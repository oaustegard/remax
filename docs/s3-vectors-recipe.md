# S3 Vectors recipe (managed ANN, float32)

Build an [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
index over a SPECTER2 corpus, using remax for the dimensional transform and
the record-ID mapping, and let AWS own the index.

This is a **recipe**, not a runtime dependency. Nothing in `remax` imports
`boto3`, and no library code was added for it.

## When this path, and when not

S3 Vectors stores `float32` only — `dataType` has exactly one valid value —
and scores with `cosine` or `euclidean`. There is no binary type and no raw
inner product.

That rules out remax's actual product. Sign-bit codes cannot be stored as
codes, so the precision ladder, the Hamming scan, and the native popcount
kernel are all unreachable here. What remains is remax's *transform*: decide
how many dimensions to keep, apply the identical transform to corpus and
query, and keep the mapping from array position to record ID.

| | Binary path | S3 Vectors path (this doc) |
|---|---|---|
| Index | remax `Corpus`, you run it | AWS runs it |
| Stored form | packed sign bits, 32 B/vector at 256-d | float32, 1024 B/vector at 256-d |
| Scoring | Hamming, native popcount | cosine or euclidean, managed |
| remax's role | the whole pipeline | the transform + ID mapping + stage 2 |
| Scaling limit | your RAM (3.2 GB at 100M × 32 B) | 2 billion vectors per index |

Take the binary path ([`docs/specter2-search-pipeline.md`](specter2-search-pipeline.md))
when you want 32 bytes per vector and control of the scan. Take this one when
you would rather not operate an index at all, and 1 KB per vector is
acceptable.

## Prerequisites

```bash
pip install remax boto3
```

Plus AWS credentials with `s3vectors:CreateIndex`, `s3vectors:PutVectors`,
and `s3vectors:QueryVectors`.

## 1. Choose the operating point

Truncate to **256 dimensions** and let the index score **cosine**. On 10,000
SPECTER2 embeddings across 8 seeds, that returns 99.8% of the true top-10
inside the top-100:

| transform | dim | metric | R@10 | R@100 |
|---|---:|---|---|---|
| truncate | 256 | cosine | **0.703 ±0.014** | **0.998 ±0.001** |
| truncate | 256 | ip | 0.413 ±0.011 | 0.887 ±0.013 |
| center+truncate | 256 | cosine | 0.661 ±0.015 | 0.987 ±0.004 |
| center+truncate | 256 | ip | 0.470 ±0.011 | 0.935 ±0.006 |

Ground truth is the top-10 by full-768-d float32 inner product; `R@N` is the
fraction of that top-10 returned in the top-N. Full sweep, per-seed spread,
and the driver: [`bench/results/S3_VECTORS_TRANSFORM.md`](../bench/results/S3_VECTORS_TRANSFORM.md).

Two things about that table are worth pausing on, because both cut against
what the rest of this repository will lead you to expect.

**Do not center on this path.** Centering is the single biggest lever on the
binary path — +0.324 R@100 at k=64 — and `Corpus.build(center=True)` exists
to make it automatic. It is a net loss here: truncate-only wins by +0.042
R@10 at 256-d, on 8 of 8 seeds. Sign bits need centering because they
threshold at the origin, so a coordinate whose mean sits far from zero
(SPECTER2 has one at ≈ 15.5) emits the same bit for nearly every record. A
float32 cosine index never thresholds, and it already discards magnitude by
normalizing; subtracting the mean only moves the origin away from the one the
ground truth is defined about. Section 2 persists the mean anyway, and shows
the A/B, because this is an encoder-specific result and yours may differ.

**The 0.943 you may have seen is an inner-product number.**
[`bench/results/SKETCH_MATRYOSHKA.md`](../bench/results/SKETCH_MATRYOSHKA.md)
measured center+truncate at 256-d with an IP scan. S3 Vectors does not offer
IP, so that figure describes a ranking the service will not compute for you.
Under cosine, the same operating point reads 0.987, and truncate-only reads
0.998. Both reproduce that table's rows exactly at its own seed — the
difference is the metric, not the harness.

Other dimensions, if 256 is not your budget:

| dim | R@100 (truncate, cosine) | bytes/vector | 100M storage |
|---:|---|---:|---|
| 128 | 0.979 ±0.007 | 512 | 51 GB |
| 192 | 0.994 ±0.002 | 768 | 77 GB |
| **256** | **0.998 ±0.001** | **1024** | **102 GB** |
| 384 | 1.000 ±0.001 | 1536 | 154 GB |
| 768 | 0.999 ±0.002 | 3072 | 307 GB |

256-d is the knee. 384-d buys the last 0.002 for 50% more storage and query
cost; 128-d gives back 0.019 to save half.

## 2. Build the corpus and persist the transform

`Corpus` is what keeps the transform reproducible: it persists the mean as
`mean.npy` and owns the position → record-ID mapping you need to resolve
query results. Build it once, offline.

```python
import numpy as np
from remax import Corpus

TRUNC = 256  # must match the index dimension, and the query transform

# corpus_vectors: (n, 768) float32, corpus_ids: list[str] aligned by index
corpus = Corpus.build(
    "my_index",
    vectors=corpus_vectors,
    ids=corpus_ids,
    seed=42,
    center=True,        # persists mean.npy; see below before you apply it
)

# Later, in the query process:
corpus = Corpus("my_index")
mean = corpus.mean            # (768,) float32, or None if built without centering
assert corpus.centered
```

`center=True` writes `mean.npy` alongside the index and exposes it as
`corpus.mean`. Persisting it is free and worth doing even if you do not
apply it: a mean you did not save is a transform you cannot reproduce, and
an index queried through a different transform than it was built with returns
silently-degraded neighbours rather than an error.

`Corpus.build` also encodes a full binary index, which this path does not
use. That is worth it when you want the binary stage 1 as well (either as a
cheaper local filter or to A/B the two paths), or want the SQLite metadata
store and `corpus.lookup(record_id)`. If you want none of that and only need
the transform reproducible, the mean is the whole of it:

```python
np.save("mean.npy", corpus_vectors.mean(axis=0))
```

Centering commutes with prefix truncation — `(x - mu)[:k] == x[:k] - mu[:k]`,
and the mean of truncated vectors is the truncation of the mean — so it does
not matter whether you build the `Corpus` at 768-d and slice the mean, or
build it at 256-d directly.

The transform itself, with centering as the flag the measurement says to
leave off:

```python
def to_index_vectors(X, mean=None, trunc=TRUNC):
    """Corpus/query → what goes into (or queries) the S3 Vectors index.

    Pass mean=None for truncate-only, the measured default. Pass
    mean=corpus.mean to center first; see bench/results/S3_VECTORS_TRANSFORM.md.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if mean is not None:
        X = X - np.asarray(mean, dtype=np.float32)
    return X[:, :trunc]
```

No normalization step: `distanceMetric="cosine"` normalizes internally, and
cosine is scale-invariant, so pre-normalizing changes the scores but not the
ranking. (If you pick `euclidean` instead, you *must* L2-normalize before
upload and at query time — euclidean on unit vectors is rank-equivalent to
cosine, and euclidean on unnormalized vectors is not.)

## 3. Create the vector bucket and index

```python
import boto3

s3vectors = boto3.client("s3vectors", region_name="us-west-2")

s3vectors.create_vector_bucket(vectorBucketName="specter2-vectors")

s3vectors.create_index(
    vectorBucketName="specter2-vectors",
    indexName="papers-256",
    dataType="float32",          # the only valid value
    dimension=TRUNC,             # 256; 1–4096 allowed
    distanceMetric="cosine",     # or "euclidean"
    metadataConfiguration={
        # Excluded from filtering; does not count against the 2 KB
        # filterable-metadata budget. Good place for display fields.
        "nonFilterableMetadataKeys": ["title"],
    },
)
```

`dimension`, `distanceMetric`, `dataType`, and `nonFilterableMetadataKeys`
are **immutable after creation**. Changing any of them means building a new
index, so settle the operating point in section 1 first.

## 4. Batch upload

`put_vectors` takes up to 500 vectors per call and a 20 MiB payload. Batch to
the limit — uploads are billed per GB **with a 128 KB minimum per request**,
so a one-vector PUT of 1 KB is charged as 128 KB. At 500 × 1 KB the minimum
stops binding entirely.

```python
import itertools

def batched(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk

index_vectors = to_index_vectors(corpus_vectors)   # (n, 256) float32

records = (
    {
        "key": rid,                                 # your record ID
        "data": {"float32": vec.tolist()},          # must be a plain list
        "metadata": {"year": 2024, "title": title},
    }
    for rid, vec, title in zip(corpus_ids, index_vectors, corpus_titles)
)

for batch in batched(records, 500):
    s3vectors.put_vectors(
        vectorBucketName="specter2-vectors",
        indexName="papers-256",
        vectors=batch,
    )
```

`key` is what comes back from a query, so make it the same string you passed
as `ids` to `Corpus.build` — that is what lets `corpus.lookup(record_id)`
resolve a result back to a position, and `Corpus.search` results back to
S3 Vectors keys.

Metadata budget per vector: 40 KB total, 50 keys, of which filterable
metadata may be 2 KB. Up to 10 non-filterable keys per index.

## 5. Query

The query goes through **the same transform as the corpus**. This is the one
step that silently returns plausible-but-wrong neighbours when you get it
wrong, since nothing validates it end to end.

```python
query_vec = encode_query("efficient methods for training large language models")
q = to_index_vectors(query_vec)[0]          # (256,) float32 — same transform

response = s3vectors.query_vectors(
    vectorBucketName="specter2-vectors",
    indexName="papers-256",
    queryVector={"float32": q.tolist()},
    topK=100,
    returnDistance=True,
    returnMetadata=True,
    # filter={"year": {"$gte": 2020}},      # optional, filterable keys only
)

candidate_ids = [v["key"] for v in response["vectors"]]
```

`topK` goes up to 10,000, but a response page holds at most 100 results —
paginate above that. Returned data is billed per GB with the first 512 KB per
query free, so `returnMetadata=True` on a wide `topK` is not free.

If your corpus was embedded with the SPECTER2 **proximity** adapter (as the
S2 public release is), encode short search queries with **`adhoc_query`**
instead — same 768-d space, trained for the asymmetric query→document
setting. See [`docs/specter2-search-pipeline.md`](specter2-search-pipeline.md)
for the encoder details.

## 6. Stage 2 rerank — and when to skip it

Re-score the candidates against full 768-d float32 inner product:

```python
candidate_vecs = lookup_full_embeddings(candidate_ids)   # (100, 768) float32
query_full = encode_query_full(query_text)               # (768,) — no truncation

scores = candidate_vecs @ query_full
top10 = [candidate_ids[i] for i in np.argsort(-scores)[:10]]
```

**Stage-1 R@100 is an exact ceiling on stage-2 R@10, and the ceiling is met.**
Reranking re-scores with the same full-768 IP that defines the ground truth,
so it orders the true top-10 correctly whenever they are in the candidate set
and cannot recover one that is absent. Measured: `rerank R@10` equals stage-1
R@100 to three decimals in every row of the sweep.

At 256-d cosine that means stage 2 turns R@100 = 0.998 into R@10 = 0.998. The
missing 0.002 is candidates stage 1 never returned, so a *better* reranker
cannot reach them — only a wider `topK` or more dimensions can.

Which makes the honest recommendation: **skip stage 2** unless you already
keep the full 768-d vectors somewhere for other reasons. It costs a second
storage system and a fetch on the query path, to reorder a candidate set
that is already 99.8% complete. The binary path needs stage 2 because its
stage 1 is much coarser; this one largely does not.

## Cost

S3 Vectors bills storage, uploads, and queries separately (US East, August
2026 — check the [pricing page](https://aws.amazon.com/s3/pricing/), these
move):

| Component | Price | Scales with dimension? |
|---|---|---|
| Storage | $0.06/GB-month | yes |
| Upload | $0.20/GB, min 128 KB per PUT | yes |
| Query requests | $2.50/million | no |
| Query data processed | tiered, by index size | yes |
| Data returned | $0.01/GB, first 512 KB/query free | no |

Charges are computed on the logical vector size — `dimension × 4 bytes` —
plus metadata and keys. So **256-d costs one third of 768-d** on storage,
upload, and data-processed, and exactly the same on the per-request fee.
At 100M papers: 102 GB and about $6/month of storage at 256-d, against
307 GB and about $18/month at 768-d.

For comparison, the binary path stores the same 100M vectors in 3.2 GB of
your own RAM. This path trades roughly 32× the bytes for not operating
anything.

## Validate before you commit to it

Every number here is 10,000 SPECTER2 embeddings — 0.01% of the S2 corpus,
one encoder. The centering result in particular is a property of SPECTER2's
geometry (norms clustered at 21.707 ± 0.146), not a law. Both immutable
choices — `dimension` and `distanceMetric` — are ones you cannot revise
without rebuilding, so measure on your own embeddings first:

```bash
python bench/s3_vectors_transform.py --dims 128 256 384 --seeds 0 1 2
```

Point it at your own cache to get the equivalent table for your encoder. If
your norms vary widely, expect the IP-vs-cosine gap to close and the
centering question to reopen.

## Limits

| | |
|---|---|
| Dimensions | 1–4096 |
| Data type | `float32` only |
| Distance metric | `cosine`, `euclidean` |
| Vectors per index | 2 billion |
| Indexes per bucket | 10,000 |
| Vectors per `put_vectors` | 500 (20 MiB payload) |
| Vectors per `get_vectors` | 100 |
| `topK` | 10,000 (100 results per response page) |
| Metadata per vector | 40 KB total, 50 keys, 2 KB filterable |
| Write throughput | 1,000 requests/s, 2,500 vectors/s per index |

Current list: [S3 Vectors limitations and restrictions](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html).
