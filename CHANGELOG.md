# Changelog

## Unreleased

Two independent lines of work: query-path throughput, and a consolidation pass
that removed about a fifth of the Python in the repository.

Everything in the throughput half is **bit-identical** to what it replaces —
the scan stays exhaustive, there is still no cell assignment, no multi-probe
and no candidate pruning, so recall is unchanged by construction rather than
by tolerance. `bench/gates/query_path_gate.py` is what holds that, and it now
proves itself against 14 simulated defects rather than 8.

The consolidation half **removes two exported functions**; see
*Removed — breaking* below.

## Unreleased — query path

### Added

- **Threaded scan.** `hamming_distances(..., threads=)`, `search(..., threads=)`,
  `packing.set_default_threads()`, and the `REMAX_THREADS` environment
  variable. The C kernel is called through `ctypes`, which releases the GIL, so
  the scan parallelises with no change to the C at all. Row blocks partition
  the corpus and each thread writes a disjoint slice of one output buffer, so
  the answer cannot depend on the thread count.

  Off by default. A library that silently spawns threads inside somebody
  else's worker pool is a bad neighbour, and — measured — threading a small
  corpus is *slower* than not threading it, so there is also no free lunch to
  hand out by default.
- **Counting select.** `packing.counting_top_k`, dispatched automatically from
  `stable_top_k` for integer distances above a measured size threshold.
  Hamming distances are integers in `[0, 8B]`, so selection needs a histogram
  rather than a comparison partition; this replaces `argpartition`'s `8n`-byte
  permutation (80 MB at n=1e7) with ~2 KB of counters. Same byte-for-byte
  contract as before — `np.argsort(kind="stable")[:k]`, the contract PR #32
  exists to defend. The float caller in `search_asymmetric` keeps the
  comparison path, where a histogram is not defined.
- **`Corpus(path, residency="mmap")`.** Maps `index.bin` read-only instead of
  reading it into private heap: O(1) open, pages faulted on touch, shared
  between processes, evictable under pressure. Default is `"load"`, unchanged.
  The memmap window is asserted C-contiguous at open — losing that would turn
  the contiguity guard added in #63 into a whole-index copy on every query
  while the open-time number still looked like a win.
- **`Corpus.search(..., asymmetric=True)`.** The measured +0.019 nDCG@10 at
  128 B/vector (+0.084 at 16 B, LFM2.5/SciFact) was previously unreachable
  through `Corpus`, which is the only supported index API — it hardcoded the
  symmetric path. Off by default because it is substantially slower.
- **`bench/query_path_speed.py`** and `bench/results/QUERY_PATH_SPEED.md` — a
  benchmark, explicitly not a gate, stating its box, its min-of-trials rule,
  what makes each pair of arms comparable, and which corpus sizes were
  actually run.

### Changed

- **The m-query loop is blocked.** `SignBitQuantizer.search` and the new
  `packing.hamming_topk_batch` read a cache-sized block of the corpus once and
  score it against all m queries, instead of reading the whole corpus once per
  query. Exact across blocks: each block contributes its own stable top-k,
  merged by a stable sort on distance, which reproduces `(distance, index)`
  order because blocks are visited in increasing row order. `block=len(codes)`
  restores the old loop; the output is identical either way.
- **`search_asymmetric` builds all m byte tables in one batched GEMM** before
  its loop rather than one small GEMM per iteration. Bit-identical.

### Notes

Three things found by the gate and the benchmark rather than by review, all
recorded where they happened:

- The first thread-count cutoff was **16384 rows**, which at B=32 is 512 KB —
  well inside the region where threading measures 0.5–0.8x, i.e. a slowdown.
  The correctness gate is structurally unable to see this: the answers were
  right the whole time. The cutoff is now 2 MB of code *per thread*, derived
  from the measured dispatch cost (~60 µs) against the ~9.7 GB/s single-core
  scan rate.
- The dropped-tail known-bad reported **ACCEPTED** twice, at thread counts
  that happened to divide the corpus size exactly — where the naive `n // T`
  split is genuinely correct and the "known-bad" is not bad at all. The gate's
  corpus size is now searched for rather than typed.
- Batched and single-query `search_asymmetric` scores already differed in the
  last ulp on v0.1.0, because `query @ rotation_` selects a different BLAS
  kernel at m=1 than at m=8. Pre-existing, unrelated to the table hoist, and
  worth knowing before someone else spends an afternoon on it.

## Unreleased — consolidation

### Removed — breaking

- **`remax.hamming_search` and `remax.asymmetric_search` are gone.** Both were
  thin wrappers the library itself declined to use: every internal caller went
  straight to `hamming_distances` + `stable_top_k`. They were exported in
  `__all__`, so this is a breaking change for anyone who imported them — it is
  the reason this section exists rather than being folded into a patch note.
  The two-line functional equivalent is recorded in `packing.py`'s module
  docstring, so the replacement is in the file where the removal happened.

  Note they shipped in **v0.1.0**, so code pinned to that tag is unaffected.

- **The benchmark harness no longer ships inside the wheel.** `src/remax/bench/`
  moved to the top-level `bench/` (git-mv, history preserved). A built wheel now
  contains eight `remax/*.py` modules and no benchmark code; `remax.bench`
  becomes a `ModuleNotFoundError` on a fresh install. Both
  `python bench/run_baseline.py` and `python -m bench.run_baseline` still work
  from a checkout. No `pyproject.toml` change was needed — `packages.find`
  scoped to `src/` is exactly what keeps a top-level `bench/` out of the wheel.

- **Ten Nemotron / NVFP4 benchmark drivers deleted** (~4,645 lines). They were
  unrunnable outside the session that wrote them. **Every conclusion is kept**:
  `bench/results/NEMOTRON_1BIT.md`, `bench/results/NEMOTRON_MASTER.md`,
  all CSVs and PNGs, and
  `fetch_nemotron_cache.sh` — which is the pointer to the raw embeddings, i.e.
  a record rather than a driver. Both markdowns gained a "Provenance" section
  mapping each surviving artifact to the script that produced it, with the git
  command to recover it.

Total: Python in the repository goes from 22,635 to 17,756 lines (−21.6%).

### Added

- **`bench/results/LFM25_LEARNED_ROTATION.md`** — a result that existed only as
  raw JSON. ITQ beats the Haar default by **+0.0200** symmetric nDCG@10 at an
  identical 128 B/vec. The sharper finding is the negative one: free-W
  straight-through achieved the **lowest quantization error of anything tested**
  (922.23 vs ITQ's 972.60) and the **worst retrieval** (−0.1805), with
  orthogonality error 98.4 — the learned map stopped being a rotation.
  **Minimizing quantization error is not the objective.** Under holdout, ITQ's
  asymmetric gain inverts (+0.0048 → −0.0042): that part was memorization.
  Not shipped, for three reasons stated in the writeup — asymmetric scoring
  buys more (+0.0248) for free and the two do not compose; the matrix is
  4.19 MB, or 32,768 vectors' worth of codes; and it would end
  data-obliviousness, since `(d, seed)` would no longer reproduce the index.

- **`bench/results/LFM25_FINETUNE.md`** — CPU fine-tuning feasibility, with the
  overfit signal made explicit: held-out nDCG peaks at **one** epoch and
  declines by three while training loss falls 8×. LoRA is only 1.37× faster
  than a full fine-tune here, because 10 of 16 LFM2 layers are convolutional
  and PEFT cannot wrap them. Two artefacts the raw JSON hides are flagged
  rather than repeated: the feasibility run's peak-RSS column is a single
  process-wide high-water mark misattributed across three rows, and the
  frozen-classifier F1 numbers come from synthetic `--demo` data.

### Changed

- `CLAUDE.md`'s anti-goals list claimed "Numba / SIMD popcount", "C/C++
  bindings" and "disk format spec" were out of scope. All three had already
  been overridden by merged code — `_native.py` compiles a C kernel with
  `-mpopcnt` at import, and `corpus.py` specifies a `RMAX` magic-byte format.
  The list is now a table of overrides with the justification each one met, so
  the document describes the project that exists.
## v0.1.0 — 2026-08-03

First tagged release. The library has been importable and useful for a while;
what it lacked was a version anyone could pin against — which is exactly how
the `rotations_` breakage below reached a downstream repository unnoticed.

### Why this release exists

`remax_kb` depends on remax as `remax @ git+https://github.com/oaustegard/remax.git`
— unpinned, tracking `main`. On 2026-08-02 a merged change made
`StackedSignBitQuantizer.rotations_` a read-only property; `remax_kb` assigns
to it in four places, and nine of its tests began failing. Nothing caught it,
because there was no released version to pin and no CI in this repository.

Both halves are now fixed: a write-through setter restored the assignment
point, and this tag gives downstreams something to depend on.

### Added

- **CI** — pytest on Python 3.9 / 3.11 / 3.13, plus a job that runs each gate
  *and* asserts it goes red under its defect flag. This repository previously
  had no `.github/` directory at all.
- **Rotation identity is recorded on disk.** `Corpus.build(..., rotation=...)`
  persists the construction to a `rotation.json` sidecar. An absent sidecar
  resolves to `haar` — never to the library default — so indexes written
  before this release keep decoding correctly no matter what the default
  becomes later. Deliberately a sidecar, not a header field: the binary header
  is untouched and the format version is unmoved, so this stays a two-way door.
- **Batch `Corpus.search`.** A 2-D query used to raise
  `TypeError: unhashable type: 'list'`; it now returns `list[list[Result]]`.
  1-D behaviour is unchanged.
- `out=` on `hamming_distances` so the per-query loop can reuse a buffer.
- Executable README tests — every `python` fence in `README.md` is run in CI.

### Fixed

- `StackedSignBitQuantizer.rotations_` accepts assignment again, via a
  write-through setter that preserves the zero-copy view's memory saving and
  its no-desync invariant.
- **All four README examples**, none of which ran: `Corpus.create` does not
  exist (it is `Corpus.build`), `hamming_distances` is module-level rather
  than a method, and its arguments were documented in the wrong order.
- `characterize()` raised `ValueError: kth(=n) out of bounds` on any corpus
  smaller than 100 vectors.
- `characterize()` no longer carries private copies of the popcount LUT and
  sign-packing helpers, so it uses the native POPCNT kernel — **28×** faster
  on its own grid shape, with byte-identical output across 3 seeds × 8
  strategies.
- SQLite connection reuse in `Corpus` (**−27% to −53%** per search), and a
  loud failure instead of a silent whole-index copy when `codes` arrives
  non-contiguous.
- `__version__` is resolved from package metadata. It had been the literal
  `"0.0.0"` through 35+ merged pull requests — including one that changed the
  emitted codes — and `remax_kb` stamps that value into every `.kb` manifest,
  so the field meant to detect a determinism violation could never fire.

### Changed

- `scipy` moved from runtime dependencies to the `dev` / `bench` extras. It
  is imported nowhere in `src/remax/`. Note this does **not** reduce the
  dependency surface downstream: `remax_kb` pins scipy itself and needs it
  transitively via `bm25s`.
- **Native speedup restated as 25–35×** (5–11 GB/s), measured, with the ratio
  varying by `n` and `d`. The code claimed 50–60× in two places and the README
  claimed 23×; all three now agree. The "within 1.3× of memcpy" claim was
  dropped rather than restated — memcpy moves 2·n·B while the scan reads n·B
  and writes 4n, so the ratio changes direction depending on an unstated
  convention.
- `bench/latency_nemotron.py` gave float32 a fully batched GEMM while giving
  remax a per-query Python loop. Matched per-query, remax wins **20.6×**;
  matched batched, **1.6×**. This is a harness correction, not a re-measurement
  — regenerating the table needs the embedding cache.

### Known limitations

- The stacked precision ladder (`k > 1`) is still not persistable by `Corpus`,
  which supports `SignBitQuantizer` only.
- All published recall numbers come from corpora of ≤10,000 vectors. The
  "100M vectors in 3.2 GB" figure in the README is an architectural
  projection, not a measurement — see issue #29.
- `haar` remains the default rotation. `rht` is available and measured at
  recall parity with a 1.5–1.8× build speedup, but the recall delta (+0.0034
  pooled) sits inside the seed spread, so this is a build-time option rather
  than a quality improvement.
