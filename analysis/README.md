# Analysis — NLP methods

Shared helpers live in [`shared/`](shared/). Each method has its own folder with a runner script, notebook, and outputs.

## Method A — Lexicon frames + emotion ([`lexicon_frames/`](lexicon_frames/))

**Idea:** Transparent dictionaries for climate *frames*, plus off-the-shelf tone lexicons.

| Signal | Tool |
|---|---|
| Frames | [`shared/frame_lexicons.py`](shared/frame_lexicons.py) — science/uncertainty, political, economic, solutions (hits per 1k tokens) |
| Polarity | VADER compound |
| Emotion | NRC fear / anger / trust / anticipation |

**Run**

```bash
python analysis/lexicon_frames/score_articles.py          # writes mediacloud/scored/
python analysis/lexicon_frames/score_articles.py --status
```

**Notebook:** [`lexicon_frames/rq_climate_narratives.ipynb`](lexicon_frames/rq_climate_narratives.ipynb)  
Monthly RQ1 trends; 90-day pre-wave media features joined to weighted CCAM outcomes; correlations + lagged/Δ OLS.

**Why keep it:** Scales to hundreds of thousands of articles; interpretable; no GPU required.

---

## Method B — Extractive BERT QA ([`bert_qa/`](bert_qa/))

**Idea:** Ask CCAM-like questions of each article with a pretrained SQuAD2 model (`deepset/roberta-base-squad2`). No fine-tuning — climate news has no span labels.

Questions cover happening, human cause, public worry, US harm, CO₂ regulation, clean energy. Answer spans are labeled affirm / deny / other / unanswered.

**Run**

```bash
python analysis/bert_qa/run_qa.py --per-year 40
python analysis/bert_qa/run_qa.py --per-year 10 --limit-years 2018-2022   # smoke
```

**Notebook:** [`bert_qa/compare_to_ccam.ipynb`](bert_qa/compare_to_ccam.ipynb)

**Outputs:** `qa_sample_articles.parquet`, `qa_answers.parquet`, `qa_yearly_affirm_rates.parquet`

**Caveat:** Affirm rates can be low — extractive spans often don’t look like survey “yes” wording.

---

## Method C — ClimateBERT classifiers ([`climatebert/`](climatebert/))

**Idea:** Use climate-domain models from [ClimateBERT](https://huggingface.co/climatebert).

Important: [`distilroberta-base-climate-f`](https://huggingface.co/climatebert/distilroberta-base-climate-f) is a **base Masked LM**, not a classifier. We run the fine-tuned heads built from that base:

| Model | Labels |
|---|---|
| `distilroberta-base-climate-sentiment` | opportunity / neutral / risk |
| `distilroberta-base-climate-commitment` | commitment/action yes/no |
| `distilroberta-base-climate-detector` | climate-related yes/no |

**Run**

```bash
python analysis/climatebert/run_classifiers.py --reuse-qa-sample
# or a fresh stratified sample:
python analysis/climatebert/run_classifiers.py --per-year 40
```

**Notebook:** [`climatebert/compare_to_ccam.ipynb`](climatebert/compare_to_ccam.ipynb)

**Rough mapping to CCAM:** risk ↔ worry/harm; commitment ↔ policy / clean-energy priority.

---

## Method D — Sentence-BERT embeddings ([`sentence_bert/`](sentence_bert/))

**Status:** Experimental / deprioritized for the RQs.

Early work embedded articles with `all-mpnet-base-v2` and clustered with UMAP + HDBSCAN. Full-corpus embedding is slow and does not directly measure framing, tone, or CCAM constructs.

Artifacts (if present): `mediacloud/embeddings/`, cells in `mediacloud/mediacloud.ipynb`.

Prefer Methods A–C for the research questions.

---

## Fast path — titles only ([`titles_only/`](titles_only/))

Re-run Methods A–C on **MediaCloud titles** (no scraped body). Full-corpus lexicon scoring in minutes; BERT QA / ClimateBERT on a stratified title sample.

```bash
python analysis/titles_only/build_corpus.py
python analysis/titles_only/score_lexicons.py
python analysis/titles_only/run_bert_qa.py --per-year 100
python analysis/titles_only/run_climatebert.py --reuse-qa-sample
```

Notebook: [`titles_only/compare_to_ccam.ipynb`](titles_only/compare_to_ccam.ipynb).  
Use as a baseline / sensitivity check; full-text methods remain stronger for final claims.

---

## Shared utilities ([`shared/`](shared/))

| File | Role |
|---|---|
| `ccam_prepare.py` | Wave dates, string-label SPSS-style recodes, weighted wave prevalences |
| `frame_lexicons.py` | Climate frame dictionaries + scoring |

Survey-only exploration (no NLP): [`../ccam/ccam_explore.ipynb`](../ccam/ccam_explore.ipynb).
