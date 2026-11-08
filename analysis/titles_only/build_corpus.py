#!/usr/bin/env python3
"""Build a titles-only corpus from MediaCloud CSV exports (no scraping)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MC_DIR = ROOT / "mediacloud"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "titles_corpus.parquet"


def log(msg: str) -> None:
    print(msg, flush=True)


def discover_csvs(mc_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in mc_dir.rglob("*-content.csv")
        if "scraped" not in p.parts and "scored" not in p.parts
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    csvs = discover_csvs(MC_DIR)
    if not csvs:
        raise FileNotFoundError(f"No *-content.csv under {MC_DIR}")

    frames = []
    for path in csvs:
        df = pd.read_csv(
            path,
            usecols=["id", "title", "publish_date", "media_name", "url"],
        )
        frames.append(df)
        log(f"  {path.relative_to(MC_DIR)}: {len(df):,}")

    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus.drop_duplicates(subset=["id"])
    corpus["title"] = corpus["title"].fillna("").astype(str).str.strip()
    corpus = corpus[corpus["title"].str.len() >= 5].copy()
    corpus["publish_date"] = pd.to_datetime(corpus["publish_date"], errors="coerce")
    corpus = corpus.dropna(subset=["publish_date"])
    corpus["year"] = corpus["publish_date"].dt.year.astype(int)
    corpus["month"] = corpus["publish_date"].dt.to_period("M").dt.to_timestamp()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(args.out, index=False)
    log(f"wrote {args.out} ({len(corpus):,} titles, {corpus['year'].min()}–{corpus['year'].max()})")


if __name__ == "__main__":
    main()
