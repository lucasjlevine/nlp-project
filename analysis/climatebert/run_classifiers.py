#!/usr/bin/env python3
"""Run ClimateBERT classifiers on a MediaCloud sample and summarize by year.

The Hub model climatebert/distilroberta-base-climate-f is a *base* language model
(Masked LM), not a classifier. Downstream ClimateBERT heads are fine-tuned from it:

  - climatebert/distilroberta-base-climate-sentiment   → opportunity / neutral / risk
  - climatebert/distilroberta-base-climate-commitment → commitment/action yes/no
  - climatebert/distilroberta-base-climate-detector   → climate-related yes/no

No additional training — inference only on a year-stratified sample.

Example:
  python analysis/climatebert/run_classifiers.py --per-year 40
  python analysis/climatebert/run_classifiers.py --reuse-qa-sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[2]
SCRAPED_DIR = ROOT / "mediacloud" / "scraped"
OUT_DIR = Path(__file__).resolve().parent
QA_SAMPLE = ROOT / "analysis" / "bert_qa" / "qa_sample_articles.parquet"
BASE_LM = "climatebert/distilroberta-base-climate-f"
SEED = 42
MAX_CHARS = 2000  # ClimateBERT is paragraph-oriented; truncate long articles

CLASSIFIERS = {
    "sentiment": "climatebert/distilroberta-base-climate-sentiment",
    "commitment": "climatebert/distilroberta-base-climate-commitment",
    "detector": "climatebert/distilroberta-base-climate-detector",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_device():
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


def load_articles() -> pd.DataFrame:
    shards = sorted(SCRAPED_DIR.glob("text_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No scraped text under {SCRAPED_DIR}")
    cols = ["id", "title", "text", "publish_date", "media_name", "url"]
    media = pd.concat([pd.read_parquet(p, columns=cols) for p in shards], ignore_index=True)
    media = media.drop_duplicates(subset=["id"])
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
    for _, g in df.groupby("year"):
        n = min(per_year, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        parts.append(g.loc[idx])
    return pd.concat(parts).sort_values("publish_date").reset_index(drop=True)


def parse_years(spec: str | None) -> tuple[int | None, int | None]:
    if not spec:
        return None, None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    y = int(spec)
    return y, y


def make_text(title: str, text: str) -> str:
    return f"{title or ''}\n\n{text or ''}".strip()[:MAX_CHARS]


def run_classifiers(sample: pd.DataFrame, device) -> pd.DataFrame:
    texts = [make_text(str(t), str(x)) for t, x in zip(sample["title"], sample["text"])]
    out = sample[["id", "publish_date", "year", "media_name", "title"]].copy()

    for short_name, model_name in CLASSIFIERS.items():
        log(f"loading {model_name}")
        clf = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            padding=True,
            max_length=512,
        )
        t0 = time.monotonic()
        preds = list(clf(texts, batch_size=8))
        log(f"  {short_name}: {len(preds)} docs in {time.monotonic() - t0:.1f}s")
        out[f"{short_name}_label"] = [p["label"] for p in preds]
        out[f"{short_name}_score"] = [float(p["score"]) for p in preds]

    # Convenient binary / multi-class indicators for time series
    out["sentiment_risk"] = (out["sentiment_label"] == "risk").astype(int)
    out["sentiment_opportunity"] = (out["sentiment_label"] == "opportunity").astype(int)
    out["sentiment_neutral"] = (out["sentiment_label"] == "neutral").astype(int)
    out["commitment_yes"] = (out["commitment_label"].str.lower() == "yes").astype(int)
    out["climate_related"] = (out["detector_label"].str.lower() == "yes").astype(int)
    return out


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)
        .agg(
            n=("id", "size"),
            climate_related_rate=("climate_related", "mean"),
            risk_rate=("sentiment_risk", "mean"),
            opportunity_rate=("sentiment_opportunity", "mean"),
            neutral_rate=("sentiment_neutral", "mean"),
            commitment_rate=("commitment_yes", "mean"),
        )
        .sort_values("year")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-year", type=int, default=40)
    p.add_argument("--limit-years", type=str, default=None)
    p.add_argument(
        "--reuse-qa-sample",
        action="store_true",
        help=f"Reuse stratified sample from {QA_SAMPLE}",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"device={device}")
    log(f"base LM (fill-mask, not run as classifier): {BASE_LM}")

    if args.reuse_qa_sample and QA_SAMPLE.exists():
        sample = pd.read_parquet(QA_SAMPLE)
        sample["publish_date"] = pd.to_datetime(sample["publish_date"], errors="coerce")
        if "year" not in sample.columns:
            sample["year"] = sample["publish_date"].dt.year
        # Need full text — join from scraped if missing
        if "text" not in sample.columns or sample["text"].isna().all():
            log("joining article text onto QA sample ids...")
            media = load_articles()[["id", "text", "title", "media_name", "publish_date", "year"]]
            sample = sample[["id"]].drop_duplicates().merge(media, on="id", how="inner")
        log(f"reusing QA sample: {len(sample):,} articles")
    else:
        year_min, year_max = parse_years(args.limit_years)
        log("loading scraped articles...")
        media = load_articles()
        sample = stratified_sample(media, args.per_year, year_min, year_max)
        log(f"sample size: {len(sample):,}")

    sample_path = OUT_DIR / "climatebert_sample_articles.parquet"
    sample.to_parquet(sample_path, index=False)

    preds = run_classifiers(sample, device)
    preds_path = OUT_DIR / "climatebert_predictions.parquet"
    preds.to_parquet(preds_path, index=False)

    yearly = yearly_summary(preds)
    yearly_path = OUT_DIR / "climatebert_yearly_rates.parquet"
    yearly.to_parquet(yearly_path, index=False)

    meta = {
        "base_lm": BASE_LM,
        "classifiers": CLASSIFIERS,
        "n_articles": int(len(sample)),
        "reuse_qa_sample": bool(args.reuse_qa_sample and QA_SAMPLE.exists()),
        "note": "climate-f is MLM base; classifiers are fine-tuned heads from that base.",
    }
    (OUT_DIR / "climatebert_run_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"wrote {preds_path}")
    log(f"wrote {yearly_path}")
    log(f"label counts:\n{preds['sentiment_label'].value_counts().to_string()}")
    log(f"yearly rates:\n{yearly.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
