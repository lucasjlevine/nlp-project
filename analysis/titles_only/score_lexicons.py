#!/usr/bin/env python3
"""Method A on titles only: climate frames + VADER + NRC over the full title corpus."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = OUT_DIR / "titles_corpus.parquet"
SCORES_PATH = OUT_DIR / "titles_lexicon_scores.parquet"
MONTHLY_PATH = OUT_DIR / "titles_lexicon_monthly.parquet"

sys.path.insert(0, str(ROOT / "analysis" / "shared"))
from frame_lexicons import FRAME_NAMES, score_frames, tokenize  # noqa: E402

NRC_EMOTIONS = ("fear", "anger", "trust", "anticipation")
_NRC = NRCLex()


def log(msg: str) -> None:
    print(msg, flush=True)


def nrc_scores(text: str) -> dict[str, float]:
    if not text.strip():
        return {e: 0.0 for e in NRC_EMOTIONS}
    _NRC.load_token_list(tokenize(text))
    freqs = getattr(_NRC, "affect_frequencies", {}) or {}
    return {e: float(freqs.get(e, 0.0)) for e in NRC_EMOTIONS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    p.add_argument("--chunksize", type=int, default=50_000)
    args = p.parse_args()

    if not args.corpus.exists():
        raise FileNotFoundError(f"Missing {args.corpus}. Run build_corpus.py first.")

    corpus = pd.read_parquet(args.corpus)
    log(f"scoring {len(corpus):,} titles...")
    analyzer = SentimentIntensityAnalyzer()

    rows: list[dict] = []
    t0 = time.monotonic()
    for i, row in enumerate(corpus.itertuples(index=False), start=1):
        title = str(row.title)
        frames = score_frames(title)
        vader = analyzer.polarity_scores(title)
        nrc = nrc_scores(title)
        dominant = max(frames, key=frames.get) if any(frames.values()) else "none"
        if dominant != "none" and frames[dominant] <= 0:
            dominant = "none"
        rows.append(
            {
                "id": row.id,
                "publish_date": row.publish_date,
                "year": row.year,
                "month": row.month,
                "media_name": row.media_name,
                "title": title,
                **{f"frame_{k}": frames[k] for k in FRAME_NAMES},
                "dominant_frame": dominant,
                "vader_compound": float(vader["compound"]),
                "vader_pos": float(vader["pos"]),
                "vader_neu": float(vader["neu"]),
                "vader_neg": float(vader["neg"]),
                **{f"nrc_{e}": nrc[e] for e in NRC_EMOTIONS},
            }
        )
        if i % args.chunksize == 0:
            rate = i / max(time.monotonic() - t0, 1e-6)
            log(f"  {i:,}/{len(corpus):,}  {rate:.0f}/s")

    scores = pd.DataFrame(rows)
    scores.to_parquet(SCORES_PATH, index=False)
    log(f"wrote {SCORES_PATH}")

    frame_cols = [f"frame_{k}" for k in FRAME_NAMES]
    tone_cols = ["vader_compound", "nrc_fear", "nrc_anger", "nrc_trust", "nrc_anticipation"]
    monthly = (
        scores.groupby("month", as_index=False)
        .agg(
            n_titles=("id", "size"),
            **{c: (c, "mean") for c in frame_cols + tone_cols},
            share_science=("dominant_frame", lambda s: (s == "science_uncertainty").mean()),
            share_political=("dominant_frame", lambda s: (s == "political").mean()),
            share_economic=("dominant_frame", lambda s: (s == "economic").mean()),
            share_solutions=("dominant_frame", lambda s: (s == "solutions").mean()),
        )
        .sort_values("month")
    )
    monthly.to_parquet(MONTHLY_PATH, index=False)
    log(f"wrote {MONTHLY_PATH} ({len(monthly):,} months)")
    log(f"done in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
