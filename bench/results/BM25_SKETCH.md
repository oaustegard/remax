## Sparse-BM25 sign-packed count-sketch — negative result

A sparse → sign-packed count-sketch path (Charikar–Chen–Farach-Colton)
was explored as a possible extension of remax's precision ladder onto
BM25-weighted inputs. The encoder, its CSR-builder, an inverted-index
streaming variant, and a BEIR/NFCorpus benchmark were all built. The
benchmark killed the thesis on both of the criteria pre-committed in
the issue, and the entire path was subsequently removed from the
library.

### Setup

NFCorpus from BEIR (3633 docs, 323 qrels-eligible queries). For each
query we computed:

- **Sketch fidelity**: R@100 of `SparseSignBitQuantizer` top-100 vs
  full `rank_bm25.BM25Okapi` top-100.
- **Relevance fidelity**: NDCG@10 vs qrels (exponential gain).

Ladder k ∈ {64, 128, 256, 512, 1024, 2048}; ablations across
centering on/off and count-sketch (signs) vs feature-hashing (no signs).
BM25 stayed at `rank_bm25` defaults (k1=1.2, b=0.75).

### Result

At the canonical config (k=256, center=off, signs=on):

| metric                | sketch | full BM25 ceiling |
|-----------------------|-------:|------------------:|
| R@100 vs BM25 top-100 |  0.019 |             1.000 |
| NDCG@10 vs qrels      |  0.010 |             0.267 |

Random-baseline R@100 for picking 100 docs from a 3633-doc corpus is
≈ 0.028. The sketch is *at* chance, not below it — across every dim
on the ladder, with or without centering, with or without signs.

Both pre-committed thesis-kill thresholds triggered:

- ❌ R@100 = 0.019 < 0.70 (fidelity)
- ❌ NDCG ratio = 0.04× < 0.50 (relevance)

### Why it failed

The failure is structural, not a tuning miss:

- **All-positive sums**: BM25 weights are non-negative. Bucket sums in
  the sketch are dominated by their largest contributors, and `sum > 0`
  is overwhelmingly True. Doc codes collapse toward a low-entropy
  attractor (mean bits-set ≈ 50/256 instead of the balanced ≈ 128/256
  that SimHash needs for meaningful Hamming distance). Centering
  rebalances the bit distribution but doesn't recover discrimination.
- **Tiny queries**: NFCorpus queries have median 5 tokens. A 5-token
  sparse query touches only ~5 buckets in a length-k buffer; the rest
  of the query code carries no signal at all. Hamming distance between
  any doc and any query then becomes a coarse 5-bit comparison
  swamped by noise from the other (k − 5) bits.
- **Cosine ≠ BM25**: Even an exact cosine over the BM25 CSR (no
  sketching) only matches `BM25Okapi.get_scores` at R@100 ≈ 0.59 on
  this corpus — the inner-product structure of BM25 disagrees with
  cosine, and the sketch by construction approximates cosine.

Going to 8-bit-per-bucket scalar quantization (Lloyd-Max / remex
flavor) would buy at most linear improvement on the bucket-side
information loss, while doing nothing about either the all-positive
or short-query problems. Not worth pursuing.

### Disposition

Removed from the library:

- `src/remax/sparse.py` — `SparseSignBitQuantizer`
- `src/remax/bm25.py` — `bm25_csr`, `bm25_query`
- Associated tests and `rank_bm25` dev dep.

This file is kept as the durable record so the experiment isn't
silently re-tried. remax stays focused on its sweet spot — dense
embedding compression via the rank-correct stacked-sign-bit ladder.

Traditional BM25 doesn't want a compressed representation anyway:
inverted indices (Lucene, FTS5, Tantivy) already store it densely
where it matters, and the natural compute pattern is sparse inner
product over a few query terms — exactly what those engines win at.
The compression argument only revives for genuinely dense or
learned-sparse (SPLADE-style) representations, which is a different
experiment.
