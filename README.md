# Climate news narratives ↔ public opinion (MediaCloud + Yale CCAM)

Research project linking **American climate news** (MediaCloud exports) to **public opinion** (Yale/George Mason *Climate Change in the American Mind*).

## Research questions

1. **RQ1:** How have framing and emotional tone in American climate change news evolved over time?
2. **RQ2:** Are changes in these narratives associated with changes in public concern, perceived agency, and support for climate policies?

## Project layout

```
project/
├── README.md                 ← you are here
├── mediacloud/               ← news data + scraping
│   ├── YYYY…/                ← MediaCloud CSV exports by period
│   ├── scrape_articles.py    ← resumable full-text scraper
│   ├── scraped/              ← article text parquet shards
│   ├── scored/               ← lexicon frame/tone scores
│   ├── embeddings/           ← experimental Sentence-BERT cache
│   └── mediacloud.ipynb      ← load exports + scrape notes
├── ccam/                     ← survey data
│   ├── CCAM_SPSS_Data_*.csv
│   └── ccam_explore.ipynb    ← survey-only EDA
└── analysis/                 ← NLP methods + RQ notebooks
    ├── README.md             ← method-by-method guide
    ├── shared/               ← CCAM recodes + frame dictionaries
    ├── lexicon_frames/       ← Method A (primary for full corpus)
    ├── bert_qa/              ← Method B (sample + extractive QA)
    ├── climatebert/          ← Method C (sample + climate classifiers)
    ├── sentence_bert/        ← Method D (experimental; deprioritized)
    └── titles_only/          ← fast path: all methods on headlines only
```

## Data flow

```text
MediaCloud CSVs ──► scrape_articles.py ──► scraped/*.parquet
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
           lexicon score_articles      BERT QA (sample)         ClimateBERT (sample)
                    │                         │                         │
                    ▼                         ▼                         ▼
              scored/*.parquet          bert_qa/*.parquet       climatebert/*.parquet
                    │                         │                         │
                    └────────────┬────────────┴────────────┬────────────┘
                                 ▼                         ▼
                    compare to CCAM waves          ccam_explore.ipynb
```

## NLP methods used (summary)

| Method | What it does | Scale | Training? | Best for |
|---|---|---|---|---|
| **A. Lexicon frames + VADER/NRC** | Keyword frame rates (science/uncertainty, political, economic, solutions) + polarity/emotion | Full scraped corpus | No | RQ1 trends; RQ2 media↔CCAM panel |
| **B. Extractive BERT QA** | Ask CCAM-like questions of articles; classify answer spans | Stratified sample (~40/year) | No (pretrained SQuAD2) | Narrative vs opinion probes |
| **C. ClimateBERT classifiers** | Risk/opportunity/neutral; commitment; climate-detector | Same sample as B | No (pretrained heads) | Domain-adapted climate labels |
| **D. Sentence-BERT + UMAP/HDBSCAN** | Embeddings + clusters | Sample / early experiment | No | Exploratory topics only — slow & weakly tied to RQs |
| **Titles-only fast path** | Re-runs A/B/C on headlines from CSVs | Full set (A) or sample (B/C) | No | Speed + coverage without scraping |

**Not used for the RQs:** fine-tuning BERT from scratch (no climate QA labels; too slow for class timelines).

### Titles-only (recommended when iterating quickly)

```bash
python analysis/titles_only/build_corpus.py
python analysis/titles_only/score_lexicons.py
python analysis/titles_only/run_bert_qa.py --per-year 100
python analysis/titles_only/run_climatebert.py --reuse-qa-sample
# open analysis/titles_only/compare_to_ccam.ipynb
```

See [`analysis/titles_only/README.md`](analysis/titles_only/README.md) for tradeoffs.

## Quick start

```bash
# 1) Survey story (no NLP required)
#    open ccam/ccam_explore.ipynb

# 2) Scrape / resume article text
python mediacloud/scrape_articles.py --status
python mediacloud/scrape_articles.py

# 3) Lexicon scoring (full corpus, resumable)
python analysis/lexicon_frames/score_articles.py --status
python analysis/lexicon_frames/score_articles.py
#    then open analysis/lexicon_frames/rq_climate_narratives.ipynb

# 4) BERT QA on a year-stratified sample
python analysis/bert_qa/run_qa.py --per-year 40
#    then open analysis/bert_qa/compare_to_ccam.ipynb

# 5) ClimateBERT on the same sample
python analysis/climatebert/run_classifiers.py --reuse-qa-sample
#    then open analysis/climatebert/compare_to_ccam.ipynb
```

Use the project `.venv` when running scripts.

## Important caveats

- **News ≠ opinion.** Model outputs describe *article text*; CCAM measures *people*. Alignments are associational.
- **Agency** is imperfect in CCAM (no clean efficacy item). We use discuss-GW and harm-immediacy as proxies.
- Large artifacts (`scraped/`, `scored/`, model outputs) are gitignored; regenerate with the commands above.
