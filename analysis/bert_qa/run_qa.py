#!/usr/bin/env python3
"""Pretrained BERT/RoBERTa extractive QA on a MediaCloud sample → CCAM-comparable series.

No fine-tuning: uses a SQuAD2 checkpoint out of the box. That matters for a course
setting — climate news has no labeled QA spans, and training from scratch would be
slow and methodologically weak.

Example:
  python analysis/bert_qa/run_qa.py --per-year 40
  python analysis/bert_qa/run_qa.py --per-year 10 --limit-years 2018-2022
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCRAPED_DIR = ROOT / "mediacloud" / "scraped"
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "deepset/roberta-base-squad2"
MAX_CONTEXT_CHARS = 2500
SEED = 42

# Questions loosely aligned to CCAM constructs (answers must be grounded in article text).
QA_ITEMS = [
    {
        "item": "happening",
        "question": "Is global warming or climate change happening?",
        "ccam_col": "happening_yes",
    },
    {
        "item": "human_cause",
        "question": "Is climate change caused mostly by human activities?",
        "ccam_col": "human_caused",
    },
    {
        "item": "public_worry",
        "question": "Are people worried about global warming?",
        "ccam_col": "worry_high",
    },
    {
        "item": "us_harm",
        "question": "Will global warming harm people in the United States?",
        "ccam_col": "harm_US_high",
    },
    {
        "item": "regulate_co2",
        "question": "Should carbon dioxide be regulated as a pollutant?",
        "ccam_col": "support_CO2_reg",
    },
    {
        "item": "clean_energy",
        "question": "Should the United States use clean or renewable energy?",
        "ccam_col": "priority_cleanenergy_high",
    },
]

AFFIRM_RE = re.compile(
    r"\b(yes|happening|is real|human|anthropogenic|caused by (humans|people|human activities)|"
    r"worried|concerned|harm|will harm|should|must|need to|support|regulate|renewable|"
    r"clean energy|solar|wind)\b",
    re.I,
)
DENY_RE = re.compile(
    r"\b(no|not happening|hoax|myth|natural(ly)?|not caused by humans|oppose|should not|"
    r"won't harm|will not harm|not worried|unconcerned)\b",
    re.I,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_articles() -> pd.DataFrame:
    shards = sorted(SCRAPED_DIR.glob("text_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No scraped text under {SCRAPED_DIR}")
    cols = ["id", "title", "text", "publish_date", "media_name", "url"]
    frames = []
    for path in shards:
        df = pd.read_parquet(path, columns=cols)
        frames.append(df)
    media = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    media["publish_date"] = pd.to_datetime(media["publish_date"], errors="coerce")
    media = media.dropna(subset=["publish_date", "text"])
    media = media[media["text"].astype(str).str.len() >= 200].copy()
    media["year"] = media["publish_date"].dt.year
    return media


def stratified_sample(media: pd.DataFrame, per_year: int, year_min: int | None, year_max: int | None) -> pd.DataFrame:
    df = media
    if year_min is not None:
        df = df[df["year"] >= year_min]
    if year_max is not None:
        df = df[df["year"] <= year_max]
    rng = np.random.default_rng(SEED)
    parts = []
    for year, g in df.groupby("year"):
        n = min(per_year, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        parts.append(g.loc[idx])
    sample = pd.concat(parts).sort_values("publish_date").reset_index(drop=True)
    return sample


def make_context(title: str, text: str) -> str:
    body = f"{title or ''}\n\n{text or ''}".strip()
    return body[:MAX_CONTEXT_CHARS]


def classify_answer(answer: str, score: float, min_score: float) -> str:
    if score < min_score or not answer or not str(answer).strip():
        return "unanswered"
    a = str(answer).strip()
    affirm = bool(AFFIRM_RE.search(a))
    deny = bool(DENY_RE.search(a))
    if affirm and not deny:
        return "affirm"
    if deny and not affirm:
        return "deny"
    if affirm and deny:
        return "mixed"
    return "other"


def answer_one(
    tokenizer: AutoTokenizer,
    model: AutoModelForQuestionAnswering,
    question: str,
    context: str,
    device: torch.device,
) -> tuple[str, float]:
    """Extractive SQuAD-style span prediction (transformers v5 has no QA pipeline)."""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    input_ids = inputs["input_ids"][0]

    # Mark context tokens (BERT: token_type_ids; RoBERTa: after second </s>).
    if "token_type_ids" in inputs:
        context_mask = inputs["token_type_ids"][0] == 1
    else:
        sep = tokenizer.sep_token_id
        sep_positions = [i for i, t in enumerate(input_ids.tolist()) if t == sep]
        ctx_start = (sep_positions[1] + 1) if len(sep_positions) >= 2 else 1
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        context_mask[ctx_start:] = True
        if tokenizer.pad_token_id is not None:
            context_mask &= input_ids != tokenizer.pad_token_id

    masked_start = start_logits.masked_fill(~context_mask, -1e9)
    masked_end = end_logits.masked_fill(~context_mask, -1e9)

    start_idx = int(torch.argmax(masked_start))
    end_idx = int(torch.argmax(masked_end))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    score = float(
        torch.softmax(masked_start, dim=-1)[start_idx]
        * torch.softmax(masked_end, dim=-1)[end_idx]
    )
    answer = tokenizer.decode(
        input_ids[start_idx : end_idx + 1], skip_special_tokens=True
    ).strip()
    return answer, score


def run_qa(
    sample: pd.DataFrame,
    model_name: str,
    device: str,
    min_score: float,
) -> pd.DataFrame:
    torch_device = torch.device(device)
    log(f"loading QA model {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(torch_device)
    model.eval()

    rows = []
    t0 = time.monotonic()
    n_jobs = len(sample) * len(QA_ITEMS)
    done = 0
    for _, art in sample.iterrows():
        context = make_context(str(art["title"]), str(art["text"]))
        for item in QA_ITEMS:
            answer, score = answer_one(
                tokenizer, model, item["question"], context, torch_device
            )
            label = classify_answer(answer, score, min_score=min_score)
            rows.append(
                {
                    "id": art["id"],
                    "publish_date": art["publish_date"],
                    "year": int(art["year"]),
                    "media_name": art["media_name"],
                    "title": art["title"],
                    "item": item["item"],
                    "question": item["question"],
                    "ccam_col": item["ccam_col"],
                    "answer": answer,
                    "score": score,
                    "label": label,
                    "affirm": int(label == "affirm"),
                }
            )
            done += 1
            if done % 100 == 0 or done == n_jobs:
                rate = done / max(time.monotonic() - t0, 1e-6)
                log(f"  {done:,}/{n_jobs:,}  {rate:.1f}/s")
    return pd.DataFrame(rows)


def parse_years(spec: str | None) -> tuple[int | None, int | None]:
    if not spec:
        return None, None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    y = int(spec)
    return y, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-year", type=int, default=40, help="Articles sampled per calendar year")
    p.add_argument("--limit-years", type=str, default=None, help="e.g. 2018-2022")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--min-score", type=float, default=0.05, help="Below this → unanswered")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    device = args.device or pick_device()
    year_min, year_max = parse_years(args.limit_years)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading scraped articles...")
    media = load_articles()
    log(f"usable articles: {len(media):,}")
    sample = stratified_sample(media, args.per_year, year_min, year_max)
    log(f"sample size: {len(sample):,} across years {sorted(sample['year'].unique())}")
    sample_path = OUT_DIR / "qa_sample_articles.parquet"
    sample.to_parquet(sample_path, index=False)

    answers = run_qa(sample, args.model, device, min_score=args.min_score)
    answers_path = OUT_DIR / "qa_answers.parquet"
    answers.to_parquet(answers_path, index=False)

    # Year-level affirm rates (BERT "survey-like" signals from news)
    yearly = (
        answers.groupby(["year", "item", "ccam_col"], as_index=False)
        .agg(
            n=("id", "size"),
            affirm_rate=("affirm", "mean"),
            mean_score=("score", "mean"),
            unanswered_rate=("label", lambda s: (s == "unanswered").mean()),
        )
        .sort_values(["item", "year"])
    )
    yearly_path = OUT_DIR / "qa_yearly_affirm_rates.parquet"
    yearly.to_parquet(yearly_path, index=False)

    meta = {
        "model": args.model,
        "device": device,
        "per_year": args.per_year,
        "year_min": year_min,
        "year_max": year_max,
        "n_articles": int(len(sample)),
        "n_qa_rows": int(len(answers)),
        "min_score": args.min_score,
        "questions": QA_ITEMS,
    }
    (OUT_DIR / "qa_run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {answers_path}")
    log(f"wrote {yearly_path}")
    log(f"affirm rates:\n{yearly.pivot(index='year', columns='item', values='affirm_rate').round(3)}")


if __name__ == "__main__":
    main()
