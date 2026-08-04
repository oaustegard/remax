# CLAUDE.md — remax briefing for Claude Code

## What this is

A small, focused library for rank-correct cosine LSH. One job: 1-bit sign-bit signatures (Charikar 2002 SimHash) with a stacked-precision ladder (k independent rotations → k bits per dim, variance ∝ 1/k).

It is **not** a Lloyd-Max library. It does not do reconstruction-quality scalar quantization. That is [remex](https://github.com/oaustegard/remex)'s job.

## Origin

In May 2026 we discovered an empirical inversion on real SPECTER2 embeddings: 1-bit Matryoshka extraction from a Lloyd-Max code beat both 2-bit and 3-bit on R@10. The 1-bit code is bit-for-bit identical to Charikar's 2002 SimHash. The 2-bit and 3-bit Lloyd-Max boundaries are MSE-optimal but rank-broken. The blog post is the canonical reference:

→ <https://muninn.austegard.com/blog/one-bit-beats-two.html>

remax is the library that exploits this finding directly: skip the broken middle entirely, scale precision by stacking sign-bit signatures.

## Differentiation from remex IVFCoarseIndex (added 2026-05-02)

remex itself now ships a SimHash-flavored mechanism (`remex.IVFCoarseIndex`, PR [#58](https://github.com/oaustegard/remex/pull/58)). It is **not** the same use of the primitive. Internalize the difference before writing code:

| | What it does | When it helps |
|---|---|---|
| **remex IVFCoarseIndex** | SimHash assigns cells; Lloyd-Max ADC scores within cells. SimHash is the *routing* layer. | Sublinear stage-1 at very large n (≥ 10M). The bottleneck it solves is bandwidth-bound flat scan. |
| **remax StackedSignBitQuantizer** | Stacked SimHash *is* the score. No cells, no Lloyd-Max. | A rank-correct precision ladder where every step is monotone (no broken middle). |

These compose orthogonally. A future architecture might use remex IVF for routing + remax stacked for in-cell scoring. v0.1.0 doesn't go there; mention it in docstrings, don't build it.

remax's value claim is **the precision ladder, not sublinear search**. Don't reinvent IVFCoarseIndex inside remax. If the work an issue describes starts to look like cell assignment / multi-probe / nprobe, stop and reread the issue.

## Architecture (as built)

```
src/remax/           # everything here ships in the wheel
├── core.py          # SignBitQuantizer (1-bit), search + search_asymmetric
├── stacked.py       # StackedSignBitQuantizer (k-stack), same two search paths
├── rotation.py      # Haar (numpy QR) and RHT (Hadamard); ROTATIONS registry
├── packing.py       # bit-packing, popcount XOR scan, asymmetric scoring, top-k
├── _native.py       # optional C popcount kernel, compiled + cached at import
├── corpus.py        # Corpus: RMAX-magic index.bin + SQLite metadata + sidecars
└── characterize.py  # encoder characterization: strategy x k sweep, recommendation

bench/               # NOT in the wheel. Harness (importable as `bench.*`) plus
                     # standalone scripts; results and their writeups in results/
tests/               # pytest; `pip install -e .` not required, src/ on path
```

Runtime dependency is numpy, and only numpy. scipy is dev/bench only — nothing
under `src/remax/` imports it.

## Key references

- **Blog post**: <https://muninn.austegard.com/blog/one-bit-beats-two.html>
- **Charikar SimHash 2002**: <https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf>
- **remex Lloyd-Max impl** (for understanding what remax is *not*): `oaustegard/remex` — see `remex/codebook.py` and `remex/core.py`
- **remex rotation impl** (Haar via QR is comparable): `oaustegard/remex` — see `remex/mojo/src/rotation.mojo` and the numpy equivalent in `remex/core.py`
- **remex bench harness** (port the structure, swap the quantizer): `oaustegard/remex` — see `bench/specter2_eval.py`, `bench/onebit_experiment.py`

## Working norms

- Single-file PRs preferred. The whole library should fit in one head.
- Each issue has a clear "Definition of Done" — meet it, no scope creep.
- Tests required. Synthetic Gaussian for unit, real embeddings for integration.
- Bench artifacts live under `bench/results/`, which is gitignored with a
  per-file whitelist in `.gitignore`. Committing a new CSV/PNG/writeup means
  adding a `!bench/results/<name>` line in the right group there.
- **A measured rejection is an asset — delete the driver, never the record.**
  When apparatus goes, its findings stay as prose in `bench/results/*.md`
  naming what was removed and why (`bench/results/BM25_SKETCH.md`, `bench/results/ROTATION_LSH.md`,
  `bench/results/LFM25_LEARNED_ROTATION.md`, the Provenance section of `bench/results/NEMOTRON_1BIT.md`).
  Better still, make it executable: `_MIN_RHT_ROUNDS = 2` in `rotation.py`
  raises and points at the measurement that set it.
- Reproduce blog post numbers in the v0.1.0 baseline as a smoke test.
- **Check the remote before every commit — a merged PR cannot take new work.**
  Long sessions merge a PR before the follow-up to it is written, and nothing
  local says so. Run `git fetch origin main && git log --oneline origin/main..HEAD`
  first. A `git push` printing `* [new branch]` for a branch you already pushed
  once means the remote branch was deleted, i.e. its PR merged — stop and check
  rather than reading past it. Recover by branching fresh from the updated
  default and cherry-picking the orphans; never force-push a merged branch to
  reuse it. (Happened twice here: #53 and #54 each merged while follow-up
  commits were still landing on their branches.)

## Anti-goals

Still binding:

- **GPU acceleration.** No CUDA, no torch in `src/`. The scan is memory-bound
  and the native kernel already streams the index at 5-11 GB/s.
- **Any reconstruction-error path.** remax does not minimize `‖x̂ − x‖`, and
  measuring quantization error is not evidence about it. `bench/results/LFM25_LEARNED_ROTATION.md`
  is the demonstration: the transform with the lowest reconstruction error in
  that run had the worst retrieval by 0.18 nDCG.
- **Lloyd-Max anything.** Use [remex](https://github.com/oaustegard/remex).
- **Sublinear search / cell assignment / multi-probe / nprobe.** That is
  `remex.IVFCoarseIndex`'s job; see the table above.

Three anti-goals were **overridden by shipped work** and the list said otherwise
until 2026-08-03. They are recorded rather than deleted, because a reader who
finds "no C bindings" in one document and `-mpopcnt` in another needs to know
which one is current:

| former anti-goal | what actually shipped | why |
|---|---|---|
| Numba / SIMD popcount | `_native.py` compiles C using `__builtin_popcountll`, adding `-mpopcnt` on x86-64/AMD64 | 25-35x over the NumPy LUT path on the same hardware; not Numba, and the fallback is still pure numpy |
| C/C++ bindings | the same file — a 47-line C source compiled by `gcc`/`cc` at first import, cached under `~/.cache/remax`, loaded via `ctypes` | zero build-time dependency and zero install-time compiler requirement: if compilation fails, `AVAILABLE` is False and the LUT path runs |
| Disk format spec | `corpus.py` defines a versioned 32-byte header (`b'RMAX'` magic, `n`, `d`, `seed`), a v0 reader for pre-magic indexes, and `mean.npy` / `rotation.json` sidecars | a stored index that cannot say which rotation encoded it returns silently-degraded neighbours; the sidecar removes the hazard, and the header stayed frozen so it is a two-way door |

The pattern to take from that table is the one that justified each override: a
new dependency was not added, and the old path still runs when the new one is
unavailable. An anti-goal here is a statement about cost, not a taboo — but
overriding one means editing this section in the same PR, not leaving the
document to contradict the code.

What has not changed is the point of the library: the rank-correct precision
ladder, and the empirical artifact behind it — a clean numpy implementation plus
the crossover plot against remex.
