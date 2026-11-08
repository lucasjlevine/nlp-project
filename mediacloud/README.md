# MediaCloud news corpus

## Contents

| Path | Description |
|---|---|
| `2008-2013/` … `2026/` | MediaCloud CSV exports (`*-content.csv`) + zip dumps |
| `scrape_articles.py` | Resumable async scraper → `scraped/` |
| `scraped/` | Full-text parquet shards + `progress.sqlite` |
| `scored/` | Lexicon frame/tone scores (from Method A) |
| `embeddings/` | Experimental Sentence-BERT cache (Method D) |
| `mediacloud.ipynb` | Load CSVs, scrape notes, early embedding cells |

## Scrape

```bash
python mediacloud/scrape_articles.py --limit 50    # smoke
python mediacloud/scrape_articles.py               # resume full run
python mediacloud/scrape_articles.py --status
python mediacloud/scrape_articles.py --retry-errors
```

Defaults: per-host delay, cross-host concurrency, parquet shards under `scraped/`.

NLP analysis of this text lives under [`../analysis/`](../analysis/).
