#!/usr/bin/env python3
"""Method B on titles only: pretrained extractive QA on a year-stratified title sample."""

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
OUT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = OUT_DIR / "titles_corpus.parquet"
DEFAULT_MODEL = "deepset/roberta-base-squad2"
SEED = 42

QA_ITEMS = [
    {"item": "happening", "question": "Is global warming or climate change happening?", "ccam_col": "happening_yes"},
    {"item": "human_cause", "question": "Is climate change caused mostly by human activities?", "ccam_col": "human_caused"},
    {"item": "public_worry", "question": "Are people worried about global warming?", "ccam_col": "worry_high"},
    {"item": "us_harm", "question": "Will global warming harm people in the United States?", "ccam_col": "harm_US_high"},
    {"item": "regulate_co2", "question": "Should carbon dioxide be regulated as a pollutant?", "ccam_col": "support_CO2_reg"},
    {"item": "clean_energy", "question": "Should the United States use clean or renewable energy?", "ccam_col": "priority_cleanenergy_high"},
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


def stratified_sample(df: pd.DataFrame, per_year: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for _, g in df.groupby("year"):
        n = min(per_year, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        parts.append(g.loc[idx])
    return pd.concat(parts).sort_values("publish_date").reset_index(drop=True)


def classify_answer(answer: str, score: float, min_score: float) -> str:
    if score < min_score or not str(answer).strip():
        return "unanswered"
    a = str(answer).strip()
    affirm, deny = bool(AFFIRM_RE.search(a)), bool(DENY_RE.search(a))
    if affirm and not deny:
        return "affirm"
    if deny and not affirm:
        return "deny"
    if affirm and deny:
        return "mixed"
    return "other"


def answer_one(tokenizer, model, question: str, context: str, device: torch.device) -> tuple[str, float]:
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    start_logits, end_logits = outputs.start_logits[0], outputs.end_logits[0]
    input_ids = inputs["input_ids"][0]

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
    answer = tokenizer.decode(input_ids[start_idx : end_idx + 1], skip_special_tokens=True).strip()
    return answer, score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-year", type=int, default=100)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--min-score", type=float, default=0.01)
    p.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    if not args.corpus.exists():
        raise FileNotFoundError(f"Missing {args.corpus}. Run build_corpus.py first.")

    corpus = pd.read_parquet(args.corpus)
    sample = stratified_sample(corpus, args.per_year)
    sample_path = OUT_DIR / "qa_sample_titles.parquet"
    sample.to_parquet(sample_path, index=False)
    log(f"sample: {len(sample):,} titles")

    device = pick_device()
    torch_device = torch.device(device)
    log(f"loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(torch_device)
    model.eval()

    rows = []
    t0 = time.monotonic()
    n_jobs = len(sample) * len(QA_ITEMS)
    done = 0
    for art in sample.itertuples(index=False):
        context = str(art.title)
        for item in QA_ITEMS:
            answer, score = answer_one(tokenizer, model, item["question"], context, torch_device)
            label = classify_answer(answer, score, args.min_score)
            rows.append(
                {
                    "id": art.id,
                    "publish_date": art.publish_date,
                    "year": int(art.year),
                    "media_name": art.media_name,
                    "title": art.title,
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
            if done % 200 == 0 or done == n_jobs:
                log(f"  {done:,}/{n_jobs:,}  {done / max(time.monotonic() - t0, 1e-6):.1f}/s")

    answers = pd.DataFrame(rows)
    answers.to_parquet(OUT_DIR / "qa_answers.parquet", index=False)
    yearly = (
        answers.groupby(["year", "item", "ccam_col"], as_index=False)
        .agg(n=("id", "size"), affirm_rate=("affirm", "mean"), mean_score=("score", "mean"))
        .sort_values(["item", "year"])
    )
    yearly.to_parquet(OUT_DIR / "qa_yearly_affirm_rates.parquet", index=False)
    (OUT_DIR / "qa_run_meta.json").write_text(
        json.dumps(
            {
                "mode": "titles_only",
                "model": args.model,
                "per_year": args.per_year,
                "n_titles": int(len(sample)),
                "device": device,
            },
            indent=2,
        )
    )
    log(f"wrote QA outputs under {OUT_DIR}")
    log(f"affirm rates:\n{yearly.pivot(index='year', columns='item', values='affirm_rate').round(3)}")


if __name__ == "__main__":
    main()
