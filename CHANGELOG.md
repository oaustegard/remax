# Changelog

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
