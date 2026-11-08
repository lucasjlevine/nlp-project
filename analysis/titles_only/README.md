# Titles-only NLP (fast path)

Use **MediaCloud article titles only** — no scraped full text.

## Is this reasonable?

**Yes, for speed and coverage**, with clear limits:

| Upside | Downside |
|---|---|
| Titles already in CSVs (~1.27M rows) — no scrape wait | ~60-char strings → thinner framing/emotion signal |
| Lexicon scoring finishes in minutes on the full set | Extractive QA often finds little span evidence in a headline |
| Same year coverage even when scrape failed (paywalls/404s) | ClimateBERT was trained on *paragraphs*; titles are out-of-domain |
| Fair A/B vs full-text methods on the same metadata | Prefer full text when you need nuance for final claims |

Treat titles-only as a **fast baseline / sensitivity check**, not a full replacement for scraped-article analyses.

## Pipeline

```bash
# 1. Build title corpus from MediaCloud CSVs
python analysis/titles_only/build_corpus.py

# 2. Method A — lexicons on ALL titles
python analysis/titles_only/score_lexicons.py

# 3. Method B — BERT QA on a year-stratified title sample
python analysis/titles_only/run_bert_qa.py --per-year 100

# 4. Method C — ClimateBERT on the same sample
python analysis/titles_only/run_climatebert.py --reuse-qa-sample

# 5. Compare to CCAM
#    open compare_to_ccam.ipynb (kernel cwd = this folder)
```

## Outputs

| File | Contents |
|---|---|
| `titles_corpus.parquet` | id, title, publish_date, media_name, url, year |
| `titles_lexicon_scores.parquet` | frame + VADER + NRC per title |
| `titles_lexicon_monthly.parquet` | monthly aggregates |
| `qa_*.parquet` | BERT-QA sample answers / yearly rates |
| `climatebert_*.parquet` | classifier preds / yearly rates |
