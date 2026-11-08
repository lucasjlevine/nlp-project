# Method D — Sentence-BERT embeddings (experimental)

**Not the primary path for RQ1/RQ2.**

## What we tried

- Model: `sentence-transformers/all-mpnet-base-v2`
- GPU embeddings of title + article text
- Cache: `mediacloud/embeddings/article_embeddings.parquet`
- Clustering: UMAP (2D) + HDBSCAN
- Notebook cells live in `mediacloud/mediacloud.ipynb` (BERT embeddings section)

## Why it was deprioritized

1. Embedding hundreds of thousands of articles is slow on-device.
2. Clusters are exploratory topics, not explicit frames/tone aligned to the RQs.
3. No direct comparison to CCAM survey items without extra labeling.

## If you revisit it

Use a stratified sample first (same idea as `analysis/bert_qa/`), cache by article `id`, and treat clusters as descriptive only.
