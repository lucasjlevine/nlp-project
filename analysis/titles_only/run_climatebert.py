#!/usr/bin/env python3
"""Method C on titles only: ClimateBERT sentiment / commitment / detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

OUT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = OUT_DIR / "titles_corpus.parquet"
QA_SAMPLE = OUT_DIR / "qa_sample_titles.parquet"
SEED = 42

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


def stratified_sample(df: pd.DataFrame, per_year: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for _, g in df.groupby("year"):
        n = min(per_year, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        parts.append(g.loc[idx])
    return pd.concat(parts).sort_values("publish_date").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-year", type=int, default=100)
    p.add_argument("--reuse-qa-sample", action="store_true")
    p.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = p.parse_args()

    if args.reuse_qa_sample and QA_SAMPLE.exists():
        sample = pd.read_parquet(QA_SAMPLE)
        log(f"reusing QA title sample: {len(sample):,}")
    else:
        if not args.corpus.exists():
            raise FileNotFoundError(f"Missing {args.corpus}. Run build_corpus.py first.")
        corpus = pd.read_parquet(args.corpus)
        sample = stratified_sample(corpus, args.per_year)
        log(f"sample: {len(sample):,} titles")

    sample.to_parquet(OUT_DIR / "climatebert_sample_titles.parquet", index=False)
    texts = sample["title"].astype(str).tolist()
    device = pick_device()
    out = sample[["id", "publish_date", "year", "media_name", "title"]].copy()

    for short_name, model_name in CLASSIFIERS.items():
        log(f"loading {model_name}")
        clf = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=128,
        )
        t0 = time.monotonic()
        preds = list(clf(texts, batch_size=32))
        log(f"  {short_name}: {len(preds)} in {time.monotonic() - t0:.1f}s")
        out[f"{short_name}_label"] = [p["label"] for p in preds]
        out[f"{short_name}_score"] = [float(p["score"]) for p in preds]

    out["sentiment_risk"] = (out["sentiment_label"] == "risk").astype(int)
    out["sentiment_opportunity"] = (out["sentiment_label"] == "opportunity").astype(int)
    out["sentiment_neutral"] = (out["sentiment_label"] == "neutral").astype(int)
    out["commitment_yes"] = (out["commitment_label"].str.lower() == "yes").astype(int)
    out["climate_related"] = (out["detector_label"].str.lower() == "yes").astype(int)

    out.to_parquet(OUT_DIR / "climatebert_predictions.parquet", index=False)
    yearly = (
        out.groupby("year", as_index=False)
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
    yearly.to_parquet(OUT_DIR / "climatebert_yearly_rates.parquet", index=False)
    (OUT_DIR / "climatebert_run_meta.json").write_text(
        json.dumps(
            {
                "mode": "titles_only",
                "classifiers": CLASSIFIERS,
                "n_titles": int(len(sample)),
                "note": "ClimateBERT was trained on paragraphs; titles are a speed tradeoff.",
            },
            indent=2,
        )
    )
    log(f"wrote ClimateBERT title outputs under {OUT_DIR}")
    log(f"yearly:\n{yearly.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
