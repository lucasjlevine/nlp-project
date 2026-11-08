#!/usr/bin/env python3
"""Score MediaCloud scraped articles with climate frames + VADER + NRC.

Streams mediacloud/scraped/text_*.parquet, checkpoints by article id,
writes mediacloud/scored/scores_XXXXX.parquet.

Example:
  python analysis/lexicon_frames/score_articles.py --limit 5000
  python analysis/lexicon_frames/score_articles.py
  python analysis/lexicon_frames/score_articles.py --status
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[2]
SCRAPED_DIR = ROOT / "mediacloud" / "scraped"
SCORED_DIR = ROOT / "mediacloud" / "scored"
DB_PATH = SCORED_DIR / "score_progress.sqlite"
SHARD_SIZE = 2000

sys.path.insert(0, str(ROOT / "analysis" / "shared"))
from frame_lexicons import score_frames, tokenize  # noqa: E402

NRC_EMOTIONS = ("fear", "anger", "trust", "anticipation")
_NRC = NRCLex()


def log(msg: str) -> None:
    print(msg, flush=True)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scored (
            id TEXT PRIMARY KEY,
            shard TEXT,
            scored_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def already_scored(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT id FROM scored")}


def next_shard_path(out_dir: Path) -> Path:
    existing = sorted(out_dir.glob("scores_*.parquet"))
    n = 0
    if existing:
        try:
            n = int(existing[-1].stem.split("_")[1]) + 1
        except ValueError:
            n = len(existing)
    return out_dir / f"scores_{n:05d}.parquet"


def nrc_scores(text: str) -> dict[str, float]:
    if not text.strip():
        return {e: 0.0 for e in NRC_EMOTIONS}
    _NRC.load_token_list(tokenize(text))
    freqs = getattr(_NRC, "affect_frequencies", {}) or {}
    return {e: float(freqs.get(e, 0.0)) for e in NRC_EMOTIONS}


def score_row(row: dict, analyzer: SentimentIntensityAnalyzer) -> dict:
    title = str(row.get("title") or "")
    body = str(row.get("text") or "")
    text = f"{title}\n{body}".strip()
    frames = score_frames(text)
    vader = analyzer.polarity_scores(text[:50000])
    nrc = nrc_scores(text[:50000])
    dominant = max(frames, key=frames.get) if any(frames.values()) else "none"
    if dominant != "none" and frames[dominant] <= 0:
        dominant = "none"
    return {
        "id": row["id"],
        "url": row.get("url"),
        "media_name": row.get("media_name"),
        "title": title,
        "publish_date": row.get("publish_date"),
        **{f"frame_{k}": v for k, v in frames.items()},
        "dominant_frame": dominant,
        "vader_compound": float(vader["compound"]),
        "vader_pos": float(vader["pos"]),
        "vader_neu": float(vader["neu"]),
        "vader_neg": float(vader["neg"]),
        **{f"nrc_{k}": v for k, v in nrc.items()},
        "n_chars": len(text),
    }


def iter_scraped_rows(scraped_dir: Path):
    shards = sorted(scraped_dir.glob("text_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No text_*.parquet under {scraped_dir}")
    for path in shards:
        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            if not row.get("id"):
                continue
            if not str(row.get("text") or "").strip():
                continue
            yield row


def print_status(conn: sqlite3.Connection) -> None:
    n = conn.execute("SELECT COUNT(*) FROM scored").fetchone()[0]
    log(f"scored articles: {n:,}")
    n_shards = len(list(SCORED_DIR.glob("scores_*.parquet")))
    log(f"score shards: {n_shards}")


def run(limit: int | None, shard_size: int) -> None:
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(DB_PATH)
    done = already_scored(conn)
    log(f"already scored: {len(done):,}")

    analyzer = SentimentIntensityAnalyzer()
    buffer: list[dict] = []
    newly = 0
    t0 = time.monotonic()

    for row in iter_scraped_rows(SCRAPED_DIR):
        if row["id"] in done:
            continue
        scored = score_row(row, analyzer)
        buffer.append(scored)
        newly += 1
        done.add(row["id"])

        if len(buffer) >= shard_size:
            path = next_shard_path(SCORED_DIR)
            pd.DataFrame(buffer).to_parquet(path, index=False)
            now = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                "INSERT OR IGNORE INTO scored (id, shard, scored_at) VALUES (?, ?, ?)",
                [(r["id"], path.name, now) for r in buffer],
            )
            conn.commit()
            log(f"  wrote {path.name} (+{len(buffer):,}; total new this run {newly:,})")
            buffer.clear()

        if limit is not None and newly >= limit:
            break

        if newly and newly % 1000 == 0:
            rate = newly / max(time.monotonic() - t0, 1e-6)
            log(f"  scored {newly:,} new @ {rate:.1f}/s")

    if buffer:
        path = next_shard_path(SCORED_DIR)
        pd.DataFrame(buffer).to_parquet(path, index=False)
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT OR IGNORE INTO scored (id, shard, scored_at) VALUES (?, ?, ?)",
            [(r["id"], path.name, now) for r in buffer],
        )
        conn.commit()
        log(f"  wrote {path.name} (+{len(buffer):,})")

    log(f"done; newly scored {newly:,}")
    print_status(conn)
    conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--status", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status(connect(DB_PATH))
        return
    run(limit=args.limit, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
