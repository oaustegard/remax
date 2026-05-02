# CLAUDE.md — remax briefing for Claude Code

## What this is

A small, focused library for rank-correct cosine LSH. One job: 1-bit sign-bit signatures (Charikar 2002 SimHash) with a stacked-precision ladder (k independent rotations → k bits per dim, variance ∝ 1/k).

It is **not** a Lloyd-Max library. It does not do reconstruction-quality scalar quantization. That is [remex](https://github.com/oaustegard/remex)'s job.

## Origin

In May 2026 we discovered an empirical inversion on real SPECTER2 embeddings: 1-bit Matryoshka extraction from a Lloyd-Max code beat both 2-bit and 3-bit on R@10. The 1-bit code is bit-for-bit identical to Charikar's 2002 SimHash. The 2-bit and 3-bit Lloyd-Max boundaries are MSE-optimal but rank-broken. The blog post is the canonical reference:

→ <https://muninn.austegard.com/blog/one-bit-beats-two.html>

remax is the library that exploits this finding directly: skip the broken middle entirely, scale precision by stacking sign-bit signatures.

## Architecture (target)

```
remax/
├── core.py          # SignBitQuantizer (1-bit), Haar rotation, pack/query
├── stacked.py       # StackedSignBitQuantizer (k-stack)
├── rotation.py      # Haar (numpy QR) and structured (Hadamard) variants
└── packing.py       # bit-packing utilities, popcount XOR scan
```

Pure-Python first. Numpy/scipy only. Numba/SIMD optimizations are non-goals for v0.1.0.

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
- Bench artifacts (CSVs, plots) live under `bench/results/` (gitignored, except `.gitkeep`).
- Reproduce blog post numbers in the v0.1.0 baseline as a smoke test.

## Anti-goals for v0.1.0

- GPU acceleration
- Numba / SIMD popcount
- C/C++ bindings
- Disk format spec
- Any reconstruction-error path
- Lloyd-Max anything (use remex)

These are not "later"; they are deliberately out of scope. v0.1.0 is the empirical artifact: a clean numpy library and the crossover plot vs remex.
