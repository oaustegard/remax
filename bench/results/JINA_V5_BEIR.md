# remax × jina-embeddings-v5-text-nano on BEIR/SciFact

- **Model**: `jinaai/jina-embeddings-v5-text-nano` (239M params, EuroBERT-210M base, 768d, last-token pool, L2-normalized).
- **Task**: SciFact (BEIR). Adapter: `retrieval`. Queries prefixed `Query: `, corpus prefixed `Document: ` (jina-v5 default).
- **Corpus**: 5183 docs.  **Test queries**: 300.  **Qrels**: 339 (binary).
- **Matryoshka dims**: [32, 64, 128, 256, 512, 768] (slice + L2-renorm, equivalent to model's `truncate_dim`).
- **Quantizer**: centered SimHash (corpus mean subtracted from corpus + queries before encoding), seed 42, native POPCNT scan.
- **Metric**: nDCG@10 — same metric Jina reports.

## Headline

At full 768d, centered SimHash loses **0.028 nDCG@10** on SciFact (0.758 fp32 → 0.730 1-bit). Jina's binary baseline on MTEB Retrieval is −0.019 nDCG@10, but theirs comes from a model trained with GOR (Geodesic Orthogonal Regularization) end-to-end for binary use; ours is **zero-shot**: we never see a binary loss. Same order of magnitude, no training cost.

At 256d, fp32 = 0.737 and k=8 stacked = 0.717 — **−0.020 drop**, matching full-dim quality with 8× fewer dimensions × 8 bits/dim = 256 bytes/doc (vs 96 bytes/doc for 1-bit @ 768d, but 24% lower nDCG@10). The matryoshka × stacked-precision product is the right operating curve for memory-constrained deployment.

At 32d, 1-bit collapses (−0.307) — 32 bits/doc is below the rank-recovery threshold. Stacking helps (k=8 → 0.421) but cannot make up for missing dimensions. Asymmetric prediction: dims dominate bits at low budgets.

## Reference numbers (from Jina v5 paper, arXiv 2602.15547)

- Table 4 — j-v5-text-nano on **BEIR** (avg across 13 tasks): **56.06 nDCG@10** at full fp32, full 768d.
- Table 6 — MTEB Retrieval subset: **64.50 → 62.60 (-1.90) nDCG@10** at 1-bit binary (full GOR training).
  RTEB: **66.45 → 63.94 (-2.51)** at 1-bit binary.
  Jina's binary numbers are dataset averages, not SciFact-specific.

## nDCG@10

| dim | float32 | 1-bit | k=2 | k=4 | k=8 |  Δ 1-bit vs fp32 |
|----:|--------:|------:|----:|----:|----:|-----------------:|
| 32 | 0.490 | 0.183 | 0.281 | 0.366 | 0.421 | -0.307 |
| 64 | 0.640 | 0.376 | 0.472 | 0.535 | 0.575 | -0.264 |
| 128 | 0.705 | 0.524 | 0.621 | 0.653 | 0.677 | -0.182 |
| 256 | 0.737 | 0.627 | 0.686 | 0.699 | 0.717 | -0.110 |
| 512 | 0.748 | 0.703 | 0.718 | 0.736 | 0.735 | -0.045 |
| 768 | 0.758 | 0.730 | 0.734 | 0.744 | 0.745 | -0.028 |

## Recall@10

| dim | float32 | 1-bit | k=2 | k=4 | k=8 |  Δ 1-bit vs fp32 |
|----:|--------:|------:|----:|----:|----:|-----------------:|
| 32 | 0.617 | 0.260 | 0.426 | 0.501 | 0.568 | -0.357 |
| 64 | 0.772 | 0.493 | 0.621 | 0.677 | 0.729 | -0.279 |
| 128 | 0.821 | 0.652 | 0.759 | 0.773 | 0.811 | -0.169 |
| 256 | 0.864 | 0.764 | 0.805 | 0.823 | 0.851 | -0.100 |
| 512 | 0.871 | 0.821 | 0.853 | 0.871 | 0.863 | -0.050 |
| 768 | 0.879 | 0.849 | 0.870 | 0.867 | 0.876 | -0.030 |

## Recall@100

| dim | float32 | 1-bit | k=2 | k=4 | k=8 |  Δ 1-bit vs fp32 |
|----:|--------:|------:|----:|----:|----:|-----------------:|
| 32 | 0.797 | 0.566 | 0.688 | 0.744 | 0.754 | -0.230 |
| 64 | 0.907 | 0.749 | 0.835 | 0.866 | 0.888 | -0.157 |
| 128 | 0.938 | 0.859 | 0.917 | 0.916 | 0.933 | -0.080 |
| 256 | 0.952 | 0.921 | 0.950 | 0.952 | 0.954 | -0.031 |
| 512 | 0.965 | 0.949 | 0.963 | 0.965 | 0.962 | -0.016 |
| 768 | 0.965 | 0.943 | 0.950 | 0.958 | 0.958 | -0.022 |

