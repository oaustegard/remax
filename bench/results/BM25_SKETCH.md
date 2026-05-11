## remax v0.0.0 — BM25 sketch on NFCorpus

## Predictions (written before running)

Per issue #36, these were committed before the bench was run:

- Sign-packed sketch at k=256 hits **R@100 ≥ 0.85** vs BM25 top-100.
- Quality plateaus near ``k ≈ log(n)/ε² ≈ 1500`` for ``n=3633`` docs
  at ``ε=0.1`` — degrades gracefully below, not a cliff.
- Centering helps less than on dense embeddings (predict: <0.05 R@100 lift).
- Count-sketch beats feature-hashing at small k; gap closes by k=1024.
- NDCG@10 vs qrels for sketch at k=256 ≥ 0.85 × NDCG@10 of full BM25.

### What kills the thesis

- ``R@100 < 0.7 at k=256`` → JL doesn't carry through for sparse-BM25-
  distributed data; investigate why (heavy-tailed weights, insufficient
  k for sparse inputs, query-side sparsity).
- ``NDCG gap > 50% at k=256`` → BM25 ranking is too sensitive to small
  perturbations for the sketch to be a useful approximation.

## Protocol

- **Library version**: remax v0.0.0
- **Corpus**: NFCorpus — 3633 docs, 323 queries with qrels, vocab 66399.
- **Sketch fidelity metric**: R@100 of sketch top-100 vs full BM25 top-100.
- **Relevance fidelity metric**: NDCG@10 vs qrels (exponential gain). Full-BM25 baseline NDCG@10 = **0.267**.
- **BM25 hyperparameters**: k1=1.2, b=0.75 (rank_bm25 defaults — issue #36 does not tune BM25 itself).
- **Tokenization**: lowercase + whitespace split, matching ``rank_bm25``'s default.
- **Encoder seed**: 42 (SparseSignBitQuantizer).
- **Production note**: real-world pipelines feed BM25 weights from Elasticsearch / Solr / FTS5 directly via ``SparseSignBitQuantizer.encode_from_postings`` (#35) — no need to materialize the CSR. The bench harness materializes for compatibility with ``rank_bm25.BM25Okapi`` as the ground truth oracle.

## Results

| k_sketch | center | signs | bytes/doc | R@100 vs BM25 | NDCG@10 vs qrels |
|---------:|:------:|:-----:|----------:|-------------:|----------------:|
|       64 |   no   |  yes  |         8 |        0.026 |           0.008 |
|       64 |   no   |  no   |         8 |        0.019 |           0.013 |
|       64 |  yes   |  yes  |         8 |        0.024 |           0.014 |
|       64 |  yes   |  no   |         8 |        0.015 |           0.009 |
|      128 |   no   |  yes  |        16 |        0.019 |           0.009 |
|      128 |   no   |  no   |        16 |        0.017 |           0.011 |
|      128 |  yes   |  yes  |        16 |        0.019 |           0.011 |
|      128 |  yes   |  no   |        16 |        0.016 |           0.013 |
|      256 |   no   |  yes  |        32 |        0.019 |           0.010 |
|      256 |   no   |  no   |        32 |        0.017 |           0.012 |
|      256 |  yes   |  yes  |        32 |        0.019 |           0.011 |
|      256 |  yes   |  no   |        32 |        0.016 |           0.012 |
|      512 |   no   |  yes  |        64 |        0.021 |           0.009 |
|      512 |   no   |  no   |        64 |        0.017 |           0.012 |
|      512 |  yes   |  yes  |        64 |        0.018 |           0.014 |
|      512 |  yes   |  no   |        64 |        0.016 |           0.012 |
|     1024 |   no   |  yes  |       128 |        0.018 |           0.009 |
|     1024 |   no   |  no   |       128 |        0.017 |           0.012 |
|     1024 |  yes   |  yes  |       128 |        0.017 |           0.014 |
|     1024 |  yes   |  no   |       128 |        0.017 |           0.012 |
|     2048 |   no   |  yes  |       256 |        0.018 |           0.010 |
|     2048 |   no   |  no   |       256 |        0.017 |           0.012 |
|     2048 |  yes   |  yes  |       256 |        0.018 |           0.015 |
|     2048 |  yes   |  no   |       256 |        0.017 |           0.012 |

## Verdict

Canonical config (k=256, center=off, signs=on): R@100 = **0.019**, NDCG@10 = **0.010** (full-BM25 ceiling 0.267, ratio 0.04×).

Against the issue's two thesis-kill thresholds:

- ❌ **R@100 = 0.019 < 0.70** — JL does not carry through for sparse-BM25-distributed data at this sketch dim. Thesis killed on fidelity.
- ❌ **NDCG ratio = 0.04× < 0.50** — BM25 ranking is too sensitive to count-sketch perturbations to be a useful approximation. Thesis killed on relevance.

Whether to ship the sparse-to-sign path follows directly from the boxes above. Stage-2 rerank (per #36's deferred follow-up) is only worth investigating if at least the fidelity box passes — the sketch must produce a useful candidate set before a rerank can recover the ranking inside it.

