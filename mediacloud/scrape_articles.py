#!/usr/bin/env python3
"""
Resumable scraper for MediaCloud article URLs.

Designed for ~1.2M URLs across many publishers:
  - never keeps full text in a single DataFrame / in RAM
  - appends to parquet shards on disk
  - SQLite checkpoint so you can stop and resume
  - per-host rate limiting to reduce bans
  - concurrent workers across different hosts

Example:
  python scrape_articles.py --limit 100          # smoke test
  python scrape_articles.py                      # full run
  python scrape_articles.py --status             # progress
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd
import trafilatura


def log(msg: str) -> None:
    print(msg, flush=True)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT
DEFAULT_OUT = ROOT / "scraped"
DEFAULT_DB = DEFAULT_OUT / "progress.sqlite"

USER_AGENT = (
    "MediaCloudExport/2.0 (+academic research; contact: local-user)"
)
DEFAULT_PER_HOST_DELAY = 1.2  # seconds between hits to the same host
DEFAULT_CONCURRENCY = 40
DEFAULT_TIMEOUT = 20.0
DEFAULT_SHARD_SIZE = 1000
MAX_TEXT_CHARS = 200_000


@dataclass
class Job:
    id: str
    url: str
    media_name: str
    title: str
    publish_date: str


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            media_name TEXT,
            title TEXT,
            publish_date TEXT,
            host TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            http_status INTEGER,
            error TEXT,
            text_len INTEGER,
            shard TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_status_host ON jobs(status, host)"
    )
    conn.commit()
    return conn


def discover_csvs(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*.csv")
        if p.name.endswith("-content.csv") and "scraped" not in p.parts
    )


def ingest_csvs(conn: sqlite3.Connection, input_dir: Path) -> int:
    """Load URL metadata into SQLite once. Safe to re-run (INSERT OR IGNORE)."""
    csvs = discover_csvs(input_dir)
    if not csvs:
        raise FileNotFoundError(f"No MediaCloud *-content.csv files under {input_dir}")

    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    for path in csvs:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            batch = []
            for row in reader:
                story_id = (row.get("id") or "").strip()
                url = (row.get("url") or "").strip()
                if not story_id or not url:
                    continue
                batch.append(
                    (
                        story_id,
                        url,
                        row.get("media_name") or "",
                        row.get("title") or "",
                        row.get("publish_date") or "",
                        host_of(url),
                        "pending",
                        now,
                    )
                )
                if len(batch) >= 5000:
                    inserted += _insert_batch(conn, batch)
                    batch.clear()
            if batch:
                inserted += _insert_batch(conn, batch)
        log(f"ingested {path.relative_to(input_dir)}")
    conn.commit()
    return inserted


def _insert_batch(conn: sqlite3.Connection, batch: list[tuple]) -> int:
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO jobs
            (id, url, media_name, title, publish_date, host, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    return conn.total_changes - before


def fetch_pending(conn: sqlite3.Connection, limit: int | None) -> list[Job]:
    """Round-robin across hosts so workers are not stuck on one publisher."""
    sql = """
        WITH ranked AS (
            SELECT id, url, media_name, title, publish_date,
                   ROW_NUMBER() OVER (PARTITION BY host ORDER BY id) AS rn
            FROM jobs
            WHERE status = 'pending'
        )
        SELECT id, url, media_name, title, publish_date
        FROM ranked
        ORDER BY rn, id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [Job(*row) for row in rows]


def mark_result(
    conn: sqlite3.Connection,
    story_id: str,
    status: str,
    *,
    http_status: int | None = None,
    error: str | None = None,
    text_len: int | None = None,
    shard: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET status = ?, http_status = ?, error = ?, text_len = ?, shard = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            status,
            http_status,
            (error or "")[:500] or None,
            text_len,
            shard,
            datetime.now(timezone.utc).isoformat(),
            story_id,
        ),
    )


def print_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY COUNT(*) DESC"
    ).fetchall()
    total = sum(n for _, n in rows) or 1
    log("progress:")
    for status, n in rows:
        log(f"  {status:12} {n:>10,}  ({100 * n / total:5.1f}%)")
    top_errors = conn.execute(
        """
        SELECT COALESCE(error, '(none)'), COUNT(*) AS n
        FROM jobs
        WHERE status = 'error'
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 10
        """
    ).fetchall()
    if top_errors:
        log("top errors:")
        for err, n in top_errors:
            log(f"  {n:>8,}  {err}")


def extract_text(html: str, url: str) -> str | None:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


class HostLimiter:
    """Serialize and space out requests per hostname."""

    def __init__(self, delay: float):
        self.delay = delay
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_ok: dict[str, float] = {}
        self._meta = asyncio.Lock()

    async def _lock_for(self, host: str) -> asyncio.Lock:
        async with self._meta:
            if host not in self._locks:
                self._locks[host] = asyncio.Lock()
            return self._locks[host]

    async def wait(self, host: str) -> None:
        lock = await self._lock_for(host)
        async with lock:
            now = time.monotonic()
            wait_for = self._next_ok.get(host, 0.0) - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._next_ok[host] = time.monotonic() + self.delay


async def scrape_one(
    client: httpx.AsyncClient,
    limiter: HostLimiter,
    job: Job,
) -> dict:
    host = host_of(job.url)
    await limiter.wait(host)
    try:
        resp = await client.get(job.url, follow_redirects=True)
        http_status = resp.status_code
        if http_status == 429 or http_status == 503:
            return {
                "id": job.id,
                "url": job.url,
                "media_name": job.media_name,
                "title": job.title,
                "publish_date": job.publish_date,
                "status": "error",
                "http_status": http_status,
                "error": f"rate_limited_{http_status}",
                "text": None,
            }
        if http_status >= 400:
            return {
                "id": job.id,
                "url": job.url,
                "media_name": job.media_name,
                "title": job.title,
                "publish_date": job.publish_date,
                "status": "error",
                "http_status": http_status,
                "error": f"http_{http_status}",
                "text": None,
            }
        text = extract_text(resp.text, str(resp.url))
        if not text:
            return {
                "id": job.id,
                "url": job.url,
                "media_name": job.media_name,
                "title": job.title,
                "publish_date": job.publish_date,
                "status": "empty",
                "http_status": http_status,
                "error": "no_extractable_text",
                "text": None,
            }
        return {
            "id": job.id,
            "url": job.url,
            "media_name": job.media_name,
            "title": job.title,
            "publish_date": job.publish_date,
            "status": "ok",
            "http_status": http_status,
            "error": None,
            "text": text,
        }
    except httpx.TimeoutException:
        return {
            "id": job.id,
            "url": job.url,
            "media_name": job.media_name,
            "title": job.title,
            "publish_date": job.publish_date,
            "status": "error",
            "http_status": None,
            "error": "timeout",
            "text": None,
        }
    except Exception as exc:  # noqa: BLE001 - keep scraper running
        return {
            "id": job.id,
            "url": job.url,
            "media_name": job.media_name,
            "title": job.title,
            "publish_date": job.publish_date,
            "status": "error",
            "http_status": None,
            "error": type(exc).__name__,
            "text": None,
        }


def next_shard_path(out_dir: Path) -> Path:
    existing = sorted(out_dir.glob("text_*.parquet"))
    n = 0
    if existing:
        try:
            n = int(existing[-1].stem.split("_")[1]) + 1
        except ValueError:
            n = len(existing)
    return out_dir / f"text_{n:05d}.parquet"


def flush_shard(out_dir: Path, rows: list[dict]) -> str:
    path = next_shard_path(out_dir)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path.name


async def worker(
    name: int,
    queue: asyncio.Queue[Job | None],
    client: httpx.AsyncClient,
    limiter: HostLimiter,
    results: asyncio.Queue[dict],
) -> None:
    while True:
        job = await queue.get()
        try:
            if job is None:
                return
            result = await scrape_one(client, limiter, job)
            await results.put(result)
        finally:
            queue.task_done()


async def run_scrape(
    *,
    db_path: Path,
    out_dir: Path,
    limit: int | None,
    concurrency: int,
    per_host_delay: float,
    timeout: float,
    shard_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    jobs = fetch_pending(conn, limit)
    if not jobs:
        log("no pending jobs")
        print_status(conn)
        conn.close()
        return

    log(f"scraping {len(jobs):,} pending jobs "
          f"(concurrency={concurrency}, per_host_delay={per_host_delay}s)")

    queue: asyncio.Queue[Job | None] = asyncio.Queue()
    results: asyncio.Queue[dict] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    limiter = HostLimiter(per_host_delay)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    buffer: list[dict] = []
    done = 0
    ok = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=concurrency + 10,
            max_keepalive_connections=concurrency,
        ),
    ) as client:
        workers = [
            asyncio.create_task(worker(i, queue, client, limiter, results))
            for i in range(concurrency)
        ]

        async def consume() -> None:
            nonlocal done, ok, buffer
            for _ in range(len(jobs)):
                result = await results.get()
                done += 1
                status = result["status"]
                text = result.pop("text")
                if status == "ok" and text:
                    ok += 1
                    buffer.append({**result, "text": text})
                    if len(buffer) >= shard_size:
                        shard = flush_shard(out_dir, buffer)
                        for row in buffer:
                            mark_result(
                                conn,
                                row["id"],
                                "ok",
                                http_status=row.get("http_status"),
                                text_len=len(row["text"]),
                                shard=shard,
                            )
                        conn.commit()
                        buffer.clear()
                        log(f"  wrote {shard}")
                else:
                    mark_result(
                        conn,
                        result["id"],
                        status,
                        http_status=result.get("http_status"),
                        error=result.get("error"),
                    )
                    if done % 50 == 0:
                        conn.commit()

                if done % 100 == 0 or done == len(jobs):
                    elapsed = max(time.monotonic() - t0, 1e-6)
                    rate = done / elapsed
                    eta_h = (len(jobs) - done) / rate / 3600
                    log(
                        f"  {done:,}/{len(jobs):,}  ok={ok:,}  "
                        f"{rate:.1f}/s  eta={eta_h:.1f}h"
                    )

        consumer = asyncio.create_task(consume())
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
        await consumer

    if buffer:
        shard = flush_shard(out_dir, buffer)
        for row in buffer:
            mark_result(
                conn,
                row["id"],
                "ok",
                http_status=row.get("http_status"),
                text_len=len(row["text"]),
                shard=shard,
            )
        conn.commit()
        log(f"  wrote {shard}")

    print_status(conn)
    conn.close()


def reset_errors(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        """
        UPDATE jobs
        SET status = 'pending', http_status = NULL, error = NULL,
            text_len = NULL, shard = NULL,
            updated_at = ?
        WHERE status IN ('error', 'empty')
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    return cur.rowcount


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Max pending jobs this run")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--per-host-delay", type=float, default=DEFAULT_PER_HOST_DELAY)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    p.add_argument("--ingest-only", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-queue previous error/empty rows as pending",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    db_path = args.db or (out_dir / "progress.sqlite")
    conn = connect(db_path)

    n_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    if n_jobs == 0 or args.ingest_only:
        log("building job queue from MediaCloud CSVs...")
        inserted = ingest_csvs(conn, args.input_dir)
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        log(f"inserted {inserted:,} new rows; queue size {total:,}")
        if args.ingest_only:
            print_status(conn)
            conn.close()
            return

    if args.retry_errors:
        n = reset_errors(conn)
        log(f"re-queued {n:,} error/empty jobs")

    if args.status:
        print_status(conn)
        conn.close()
        return

    conn.close()
    asyncio.run(
        run_scrape(
            db_path=db_path,
            out_dir=out_dir,
            limit=args.limit,
            concurrency=args.concurrency,
            per_host_delay=args.per_host_delay,
            timeout=args.timeout,
            shard_size=args.shard_size,
        )
    )


if __name__ == "__main__":
    main()
