# remax

**Rank-correct cosine LSH with a stacked-precision ladder.**

A focused library for one job: 1-bit cosine quantization that preserves *rank*, with a precision ladder that scales linearly in storage and monotonically in recall. Built as a deliberate counterpoint to MSE-optimal scalar quantization, motivated by an empirical inversion observed on real SPECTER2 embeddings.

The name is a pun. Lloyd-Max minimizes reconstruction error. remax targets rank.

## Why this exists

In [*One Bit Beats Two*](https://muninn.austegard.com/blog/one-bit-beats-two.html), 1-bit Matryoshka extraction from a Lloyd-Max code beat both 2-bit and 3-bit on R@10 — by 13 and 4 percentage points respectively — on real SPECTER2 embeddings. The 1-bit code is bit-for-bit identical to Charikar's 2002 SimHash. Its 2- and 3-bit cousins inherit Lloyd-Max's MSE-optimal interior boundaries, which are wrong for ranking: an interior bin flip changes the dot-product sign without proportional MSE penalty.

The fix Lloyd-Max can't deliver: don't refine each coordinate, *stack independent SimHashes*. k stacked sign-bit signatures give k bits per dimension with rank-correct semantics at every step (variance ∝ 1/k), and no broken middle.

remax is the library that does that, exclusively.

## Status

**Proof-of-concept ready.** Core quantizers, native acceleration, corpus management, benchmarks, and a two-stage rerank pipeline are implemented and tested. The API is usable for real workloads — 10k-scale corpora today, with the architecture designed for 100M+ vectors (3.2 GB RAM at 32 bytes/vector).

Future work is tracked in [issues](https://github.com/oaustegard/remax/issues). The strategic direction ([#12](https://github.com/oaustegard/remax/issues/12)) targets Semantic Scholar's 220M-paper corpus via S3 Vectors + Athena.

### What's implemented

**Core library** (`src/remax/`):
- `SignBitQuantizer` — 1-bit Charikar/SimHash with corpus-mean centering. Centering is the single biggest lever: +0.324 R@100 at k=64 on SPECTER2.
- `StackedSignBitQuantizer` — k-stack precision ladder (k=2,4,8 tested). Every step shrinks variance by 1/k while remaining rank-correct. No broken middle.
- `Corpus` — packed binary codes + SQLite metadata sidecar. Maps array indices to record IDs with JSON metadata per record. [Postgres recipe](docs/postgres-recipe.md) included.
- `characterize()` — sweep a strategy × k grid on your encoder and get a recommended operating point.
- Rotation choice — `rotation="haar"` (default, Haar-distributed QR) or `rotation="rht"` (randomized Hadamard, 1.5–1.8× faster to build). Measured equivalent for retrieval; see [`ROTATION_LSH.md`](bench/results/ROTATION_LSH.md) for why a structured rotation needed re-measuring here rather than inheriting remex's result, and for the single-round construction it rules out.
- Native Hamming scan — C extension compiled at first import with hardware POPCNT. **25–35× over the NumPy LUT fallback**, 5–11 GB/s effective throughput. The ratio depends on `n` and `d` — 33× at n=1M/d=768, 28× at n=1M/d=256 — because the NumPy path materialises a uint16 gather ~3× the size of the index while the native path streams it once. Throughput drops from ~10 GB/s to ~7 GB/s between n=100k and n=1M, where the index outgrows last-level cache. Measured on an Intel Xeon @ 2.80 GHz; reproduce with `python3 bench/native_speedup.py`.

**Benchmark suite** (`bench/`):
- [`BASELINE.md`](bench/results/BASELINE.md) — R@10 vs float32 ground truth across the stacked precision ladder. 1-bit: 0.635, k=2: 0.676, k=4: 0.706, k=8: 0.718.
- [`CROSSOVER.md`](bench/results/CROSSOVER.md) — side-by-side R@10 of remax stacked SimHash vs remex Lloyd-Max at matched bits-per-dim.
- [`RERANK.md`](bench/results/RERANK.md) — two-stage pipeline: sign-bit stage 1 → float32-IP rerank recovers to R@10 = 0.983 at 0.1 ms/query. Cross-encoder rerank (ms-marco-MiniLM-L-6-v2, ONNX Runtime) tested and characterized.
- [`ROTATION_LSH.md`](bench/results/ROTATION_LSH.md) — does the Charikar collision guarantee survive a structured rotation? Collision rate vs `θ/π`, estimator spread, cross-stack independence, then end-to-end recall. Also documents two optimizations measured and rejected.
- [`SKETCH_MATRYOSHKA.md`](bench/results/SKETCH_MATRYOSHKA.md) / [`SKETCH_MATRYOSHKA_GEMINI.md`](bench/results/SKETCH_MATRYOSHKA_GEMINI.md) — post-hoc Matryoshka via random-dimension sketching on SPECTER2 and Gemini embeddings.

**Test suite**: full coverage across core, stacked, corpus, native, characterize, and all benchmark modules. Security hardening pass completed.

## Quick start

```bash
pip install -e .
```

Every `python` block in this file is executed verbatim by
`tests/test_readme.py`, so what follows runs as written — swap in your own
embeddings for the synthetic ones.

```python
import numpy as np
import remax

rng = np.random.default_rng(0)
embeddings = rng.standard_normal((1000, 768), dtype=np.float32)
query = rng.standard_normal(768, dtype=np.float32)
paper_ids = [f"paper-{i}" for i in range(len(embeddings))]

# Encode: 768 float32 dims (3 KB/vector) → 96 packed bytes
q = remax.SignBitQuantizer(d=768, seed=42)
codes = q.encode(embeddings)                     # (1000, 96) uint8

# Search: top-k by Hamming distance
top_k, dists = q.search(query, codes, k=10, return_distances=True)

# The functional API underneath, when you want the whole distance vector.
# Corpus first, query second — and the result is 1-D over the corpus.
all_dists = remax.hamming_distances(codes, q.encode(query))   # (1000,) int32

# Stacked precision ladder: k independent rotations, k bits per dimension
sq = remax.StackedSignBitQuantizer(d=768, k=4, seed=42)
stacked_codes = sq.encode(embeddings)            # (1000, 384) uint8 — 4× wider
stacked_top_k = sq.search(query, stacked_codes, k=10)

# Corpus with metadata. build() takes a *directory*, not a file, and
# quantizes internally — it does not accept a quantizer.
corpus = remax.Corpus.build(
    "papers/", embeddings, paper_ids,
    seed=42,
    meta=[{"title": f"Paper {i}"} for i in range(len(embeddings))],
    center=True,          # stores the corpus mean; search() re-applies it
)
results = corpus.search(query, k=10)             # list[Result]
print(results[0].rank, results[0].record_id, results[0].distance, results[0].meta)
```

Native POPCNT acceleration is automatic when available (check `remax.NATIVE_AVAILABLE`).

### Scaling the query path

The scan is exhaustive and stays exhaustive — remax has no cells, no probes and
no candidate pruning, so recall is a property of the code rather than of a
parameter you can get wrong. What *is* tunable is how fast the same answers
arrive. Every option below is **bit-identical** to its default; none of them
can change a neighbour.

```python
# The C kernel is reached through ctypes, which releases the GIL, so the scan
# threads with no change to the C. Off by default — a library that spawns
# threads inside your worker pool is a bad neighbour. Row blocks partition the
# corpus, so every thread count gives the identical answer.
top_k_threaded = q.search(query, codes, k=10, threads="auto")

# Process-wide, if you would rather not pass it at every call site:
remax.packing.set_default_threads(1)   # "auto" for all cores; REMAX_THREADS too

# Threading a small corpus measures SLOWER than not threading it, so remax
# bypasses the pool below a measured size rather than trusting the argument.

# A batch reads the corpus once per cache-sized block for all m queries
# instead of once per query. Automatic for m >= 2.
batch_top_k = q.search(embeddings[:8], codes, k=10)

# A single query blocks too, for a different reason. Once k candidates are
# held, their k-th distance bounds everything that could still matter, so each
# block is filtered against it while still in cache instead of every distance
# being written out and ranked in full. Selection, not the scan, is what costs
# at scale: at n=1e8 it was 0.96x the scan and is now 0.18x. Automatic above a
# measured n; see bench/results/SCALE_100M.md.
one_query = q.search(query, codes, k=10)

# mmap an index instead of reading it into private heap: O(1) open, pages
# faulted in on touch, shared between processes, evictable under memory
# pressure. The default is still "load" — the historical behaviour.
with remax.Corpus("papers/", residency="mmap") as mapped:
    mapped_results = mapped.search(query, k=10)

# Asymmetric scoring: keep the query in float against the stored sign bits.
# Same index, same bytes on disk; a query occupies no index storage, so
# binarizing it buys nothing. Worth +0.019 nDCG@10 at 128 B/vector on
# LFM2.5/SciFact, +0.084 at 16 B. Substantially slower — it cannot use the
# popcount kernel — so it is opt-in rather than the default.
better = corpus.search(query, k=10, asymmetric=True)
```

Measurements, with the box and the scope limits stated, are in
[`bench/results/QUERY_PATH_SPEED.md`](bench/results/QUERY_PATH_SPEED.md) up to
n=1e7 and [`bench/results/SCALE_100M.md`](bench/results/SCALE_100M.md) from
there to n=1e8 — two files because the first establishes nothing above 1e7 and
says so, and a bandwidth-bound scan changes regime at every cache boundary.
That the answers do not change is not measured but *gated*:
`bench/gates/query_path_gate.py`, which proves itself by going red under 16
simulated defects.

## Relationship to remex

[remex](https://github.com/oaustegard/remex) is the multi-precision Lloyd-Max + Matryoshka library it shares lineage with. remex is a Swiss Army knife optimized for storage MSE, with rank-correct 1-bit *as a free MSB extraction*. remax is a chisel optimized for rank, exclusively.

The two coexist:
- Use **remex** when you need a single 8-bit storage tier with cheap Matryoshka extraction down to 1 bit.
- Use **remax** when you need a pure-rank in-memory tier with a precision ladder that doesn't break in the middle.
- Use **both** if your two-stage retrieval architecture wants a remax-ladder Stage 1 and remex Stage 2.

## Related work: training-time quantization-friendly embeddings

Two training-time techniques converge on the property remax operates on at inference: *embeddings whose distribution shape makes 1-bit quantization rank-preserving*.

**Matryoshka Representation Learning** ([Kusupati et al. 2022](https://arxiv.org/abs/2205.13147)) trains so any prefix of an embedding is a usable embedding. This gives a **dimension knob**: truncate to 256-d or 128-d and rank is largely preserved.

**Global Orthogonal Regularizer (GOR)** ([Zhang et al. 2017](https://arxiv.org/abs/1708.06320), revived in [embeddinggemma](https://arxiv.org/abs/2509.20354) and [jina-embeddings-v5](https://arxiv.org/abs/2602.15547)) adds a contrastive term that penalizes squared cosine between non-matching pairs, pushing the distribution toward uniform-on-sphere. This gives a **precision knob**: binarize and rank is largely preserved.

Jina v5's Table 6 quantifies the GOR effect at full dimension:

| Configuration | MTEB BF16 | MTEB Binary  | RTEB BF16 | RTEB Binary  |
|---------------|-----------|--------------|-----------|--------------|
| With GOR      | 64.50     | 62.60 (−1.90) | 66.45    | 63.94 (−2.51) |
| Without GOR   | 64.21     | 61.13 (−3.08) | 66.16    | 62.24 (−3.92) |

GOR halves the binary-quantization loss at negligible BF16 cost (+0.29). The two knobs compound for in-memory first-pass retrieval — a 1024-d BF16 embedding (2 KB) becomes 32 bytes at 256-d binary (64× compression), with both operations independently rank-preserving.

The jina-v5 paper presents each knob in isolation; it does not publish recall numbers for combined truncation + binarization — the actual first-pass configuration most self-hosters would deploy. The combined story (Matryoshka + GOR) is implicit in the architecture but undersold in the evaluation.

### What this means for remax

remax is encoder-agnostic — it consumes any numpy array of embeddings and exposes a **third rank-preserving knob**: a stacked-precision ladder that gives k bits per dimension with variance ∝ 1/k. The competitive frame depends on the embedding source:

- **Already centered** (GOR-trained, or pre-normalized via other means): remax's corpus-mean centering is near-no-op. The stacked ladder remains orthogonal to both Matryoshka and GOR — pick a dimension via Matryoshka, pick a precision tier via the ladder. The ~2-point binary headroom that remains on GOR-trained models is the ladder's addressable problem.
- **Not centered** (the common case for precomputed embedding corpora today, including the published SPECTER2 artifacts): corpus-mean centering does substantial work — the +0.324 R@100 measured on SPECTER2 is the upper end of what's recoverable — and the stacked ladder layers on top.

SPECTER2 is remax's primary benchmark substrate because it's a large, publicly-available precomputed embedding corpus, not because remax targets SPECTER2 specifically. The library applies to any embedding source where the distribution shape leaves recall on the table.

## Background

This library emerged from a series of experiments documented on [muninn.austegard.com](https://muninn.austegard.com):
1. [One Bit Beats Two](https://muninn.austegard.com/blog/one-bit-beats-two.html) — the empirical inversion that started it
2. [Embedding Compression Is Mostly Centering](https://muninn.austegard.com/blog/embedding-compression-is-mostly-centering.html) — why centering matters more than rotation
3. [Three Gigs to Search a Hundred Million Papers](https://muninn.austegard.com/blog/three-gigs-to-search-a-hundred-million-papers.html) — scaling projections and random-dim sketching
4. [Matryoshka Doesn't Buy You Sign-Bit Compression](https://muninn.austegard.com/blog/matryoshka-doesnt-buy-you-sign-bit-compression.html) — why post-hoc Matryoshka doesn't help at 1-bit

## License

MIT, © 2026 Oskar Austegard.
