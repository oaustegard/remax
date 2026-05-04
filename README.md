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
- Native Hamming scan — C extension compiled at first import with hardware POPCNT. 23× over NumPy (9.7 GB/s effective throughput, within 1.3× of memcpy ceiling).

**Benchmark suite** (`bench/`):
- [`BASELINE.md`](bench/results/BASELINE.md) — R@10 vs float32 ground truth across the stacked precision ladder. 1-bit: 0.635, k=2: 0.676, k=4: 0.706, k=8: 0.718.
- [`CROSSOVER.md`](bench/results/CROSSOVER.md) — side-by-side R@10 of remax stacked SimHash vs remex Lloyd-Max at matched bits-per-dim.
- [`RERANK.md`](bench/results/RERANK.md) — two-stage pipeline: sign-bit stage 1 → float32-IP rerank recovers to R@10 = 0.983 at 0.1 ms/query. Cross-encoder rerank (ms-marco-MiniLM-L-6-v2, ONNX Runtime) tested and characterized.
- [`SKETCH_MATRYOSHKA.md`](bench/results/SKETCH_MATRYOSHKA.md) / [`SKETCH_MATRYOSHKA_GEMINI.md`](bench/results/SKETCH_MATRYOSHKA_GEMINI.md) — post-hoc Matryoshka via random-dimension sketching on SPECTER2 and Gemini embeddings.

**Test suite**: full coverage across core, stacked, corpus, native, characterize, and all benchmark modules. Security hardening pass completed.

## Quick start

```bash
pip install -e .
```

```python
import numpy as np
import remax

# Encode
q = remax.SignBitQuantizer(d=768, seed=42)
codes = q.encode(embeddings)          # (n, 96) uint8

# Search
dists = remax.hamming_distances(q.encode(query), codes)
top_k = np.argsort(dists[0])[:10]

# Stacked precision ladder
sq = remax.StackedSignBitQuantizer(d=768, k=4, seed=42)
codes = sq.encode(embeddings)         # (n, 384) uint8 — 4× wider, rank-correct
dists = sq.hamming_distances(sq.encode(query), codes)

# Corpus with metadata
corpus = remax.Corpus.create("papers.bin", embeddings, sq,
                              record_ids=paper_ids,
                              metadata=[{"title": t} for t in titles])
results = corpus.search(query_embedding, k=10)  # List[Result]
```

Native POPCNT acceleration is automatic when available (check `remax.NATIVE_AVAILABLE`).

## Relationship to remex

[remex](https://github.com/oaustegard/remex) is the multi-precision Lloyd-Max + Matryoshka library it shares lineage with. remex is a Swiss Army knife optimized for storage MSE, with rank-correct 1-bit *as a free MSB extraction*. remax is a chisel optimized for rank, exclusively.

The two coexist:
- Use **remex** when you need a single 8-bit storage tier with cheap Matryoshka extraction down to 1 bit.
- Use **remax** when you need a pure-rank in-memory tier with a precision ladder that doesn't break in the middle.
- Use **both** if your two-stage retrieval architecture wants a remax-ladder Stage 1 and remex Stage 2.

## Background

This library emerged from a series of experiments documented on [muninn.austegard.com](https://muninn.austegard.com):
1. [One Bit Beats Two](https://muninn.austegard.com/blog/one-bit-beats-two.html) — the empirical inversion that started it
2. [Embedding Compression Is Mostly Centering](https://muninn.austegard.com/blog/embedding-compression-is-mostly-centering.html) — why centering matters more than rotation
3. [Three Gigs to Search a Hundred Million Papers](https://muninn.austegard.com/blog/three-gigs-to-search-a-hundred-million-papers.html) — scaling projections and random-dim sketching
4. [Matryoshka Doesn't Buy You Sign-Bit Compression](https://muninn.austegard.com/blog/matryoshka-doesnt-buy-you-sign-bit-compression.html) — why post-hoc Matryoshka doesn't help at 1-bit

## License

MIT, © 2026 Oskar Austegard.
