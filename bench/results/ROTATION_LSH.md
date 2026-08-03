# Structured rotations for SimHash — does the collision guarantee survive?

Investigation for [#59](https://github.com/oaustegard/remax/issues/59).
Reproduce with `bench/rotation_lsh_fidelity.py`.

## The question

[remex#71](https://github.com/oaustegard/remex/pull/71) replaced a Haar
rotation with a randomized Hadamard transform (RHT) and measured a large
construction speedup at no recall cost. **That argument does not transfer to
remax**, and that was the whole reason to file this as an investigation
rather than a change.

remex needed an orthogonal map that makes coordinates ~N(0, 1/d) so its
Lloyd-Max codebook quantizes the distribution it was fitted to. Reconstruction
MSE is indifferent to *which* isotropic rotation you use, so the substitution
was safe by construction.

remax rests on something stronger — the Charikar (2002) / Goemans–Williamson
(1995) collision bound `P[sign mismatch] = θ/π`, which is a statement about
the *distribution* of the projection directions. An RHT carries `O(d log d)`
bits of randomness against Haar's `O(d²)`, and its rows are structured rather
than independent. "It worked for the reconstruction codec" is not evidence
about an LSH collision bound.

So it was measured, theory first, recall last.

## Verdict

The substitution holds — **but only with at least two mixing rounds**, which
is *not* what a direct port of remex#71 gives you. remex uses a single round
whenever `d` is a power of two. At one round the `k` rotations of a stack stop
being independent on anisotropic input, and `d ∈ {512, 1024, 2048, 4096}` —
where that branch fires — covers most mainstream embedding widths.

`rht_rotation` therefore floors `rounds` at 2 and rejects lower values.

`"haar"` stays the default: the two constructions give different codes from
the same `(d, k, seed)`, and the on-disk index header has no field recording
which was used, so flipping the default would silently invalidate every stored
corpus. See "Not done here" below.

---

## Step 1 — collision rate vs the published curve

Anchored on `θ/π` itself, not on "RHT agrees with Haar" — two equally
structured rotations agreeing with each other would prove nothing.

`d=1024`, 400 pairs × 3 rotation seeds, isotropic Gaussian input. Cell is
`mean ± sd` of the pooled collision fraction, then bias vs theory.

| θ/π | rot | k=1 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| 0.10 | haar | 0.1000±0.0087 +0.0000 | 0.0999±0.0045 −0.0001 | 0.1000±0.0032 +0.0000 | 0.1001±0.0023 +0.0001 |
| 0.10 | rht  | 0.1005±0.0089 +0.0005 | 0.1001±0.0044 +0.0001 | 0.1000±0.0031 −0.0000 | 0.1000±0.0022 +0.0000 |
| 0.35 | haar | 0.3503±0.0124 +0.0003 | 0.3501±0.0065 +0.0001 | 0.3503±0.0043 +0.0003 | 0.3501±0.0030 +0.0001 |
| 0.35 | rht  | 0.3496±0.0119 −0.0004 | 0.3499±0.0058 −0.0001 | 0.3499±0.0042 −0.0001 | 0.3499±0.0028 −0.0001 |
| 0.50 | haar | 0.4996±0.0122 −0.0004 | 0.5000±0.0062 +0.0000 | 0.5001±0.0042 +0.0001 | 0.5001±0.0030 +0.0001 |
| 0.50 | rht  | 0.5005±0.0127 +0.0005 | 0.5000±0.0063 −0.0000 | 0.5000±0.0043 +0.0000 | 0.5002±0.0030 +0.0002 |

Both track `θ/π` to within ±0.0005. Same result on heavy-tailed ("spiky")
coordinates. **Step 1 passes for both.**

## Steps 2 + 3 — spread, and whether the stacks are actually independent

This is where it breaks, and only on anisotropic input.

- `sd_ratio` = observed spread ÷ binomial `sqrt(p(1−p)/(k·d))`. Below 1 means
  the orthogonal projection directions beat genuinely independent ones —
  which is the *correct* behaviour and something `core.py`'s variance claim
  understates for Haar.
- `var_ratio` = `Var(pooled) ÷ (Var(one stack) / k)`. **1.0 means the `k`
  stacks are independent**, which is the assumption the precision ladder
  rests on.

`d=1024`, `k=8`, anisotropic input (axis-aligned variance decay plus a shared
mean direction — the regime real embeddings live in):

| θ/π | rot | bias | sd_ratio | mean \|corr\| | var_ratio |
|---|---|---|---|---|---|
| 0.20 | haar   | −0.0004 | 0.894 | 0.0450 | 1.020 |
| 0.20 | rht-r1 | **+0.0017** | 1.020 | 0.0598 | **1.306** |
| 0.20 | rht-r2 | −0.0006 | 0.871 | 0.0352 | 0.979 |
| 0.35 | haar   | −0.0014 | 0.824 | 0.0442 | 1.055 |
| 0.35 | rht-r1 | **+0.0026** | 1.014 | 0.0792 | **1.543** |
| 0.35 | rht-r2 | −0.0008 | 0.771 | 0.0377 | 0.959 |
| 0.50 | haar   | +0.0005 | 0.775 | 0.0381 | 1.033 |
| 0.50 | rht-r1 | **+0.0032** | 1.011 | **0.0900** | **1.629** |
| 0.50 | rht-r2 | +0.0004 | 0.753 | 0.0372 | 0.984 |

Three things go wrong at one round, all at once:

1. **The bias converges to a non-zero value instead of to zero.** Sampling
   error must shrink as `k` grows; a defect shared by every rotation in the
   stack cannot, because averaging `k` copies of the same structural bias
   leaves it intact. Bias vs `θ/π` at `d=1024`, anisotropic input, 800 pairs
   × 4 seeds (step 1b in the bench):

   | θ/π | rot | k=1 | k=4 | k=8 | k=16 |
   |---|---|---|---|---|---|
   | 0.35 | haar | +0.0012 | −0.0004 | +0.0001 | −0.0001 |
   | 0.35 | rht-r1 | +0.0026 | +0.0013 | +0.0011 | **+0.0019** |
   | 0.35 | rht-r2 | −0.0013 | −0.0014 | −0.0007 | −0.0003 |
   | 0.50 | haar | +0.0002 | −0.0001 | −0.0002 | +0.0002 |
   | 0.50 | rht-r1 | +0.0014 | +0.0026 | +0.0032 | **+0.0034** |
   | 0.50 | rht-r2 | +0.0002 | +0.0003 | +0.0007 | +0.0007 |

   Haar hovers at ±0.0002 at every `k`. Single-round RHT *settles onto*
   +0.0034 as `k` rises — the sampling noise averages out and exposes the
   structural offset underneath. More stacking makes it more visible, not
   less.
2. **Stack independence fails.** `var_ratio` 1.63 means a `k=8` stack
   delivered the variance of `k≈4.9`. At `d=512` the same measurement gives
   1.77 — `k≈4.5`. Roughly half the precision ladder, silently lost.
3. **The sub-binomial benefit disappears.** Haar sits at `sd_ratio` 0.78;
   single-round RHT sits at 1.01, having given up what orthogonal directions
   were buying.

`SeedSequence` gives independent *seeds* — that was never the assumption. The
assumption is independence of the resulting estimators, and with a structured
transform those are different claims.

**Two rounds fix all three**, landing on Haar's numbers across the board.
Three rounds add nothing measurable. On isotropic Gaussian input even one
round is fine, which is exactly why this needed the anisotropic case to show
up at all.

Cost of the extra round: ~0.02 s at `d=768`, ~0.8 s at `d=3072`. The guard is
close to free.

## Step 4 — end-to-end recall on real embeddings

recall@10 against exact float32 cosine ground truth. 3 seeds, `mean ± sd`.
`build` is the whole `k ∈ {1,2,4,8}` ladder (15 rotations).

| corpus | rot | build | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|
| jina-v5-nano/SciFact `d=768` | haar | 0.94 s | 0.6794±0.0062 | 0.7616±0.0031 | 0.8256±0.0054 | 0.8762±0.0058 |
| | rht | 0.57 s | 0.6803±0.0042 | 0.7641±0.0038 | 0.8324±0.0051 | 0.8776±0.0045 |
| | **Δ** | **1.66×** | +0.0009 | +0.0026 | +0.0069 | +0.0013 |
| Gemini `d=1024` | haar | 1.90 s | 0.5744±0.0035 | 0.6758±0.0042 | 0.7677±0.0026 | 0.8319±0.0022 |
| | rht | 1.26 s | 0.5670±0.0130 | 0.6751±0.0029 | 0.7689±0.0038 | 0.8269±0.0039 |
| | **Δ** | **1.51×** | −0.0074 | −0.0007 | +0.0012 | −0.0050 |
| Gemini `d=3072` | haar | 50.31 s | 0.7054±0.0019 | 0.7833±0.0020 | 0.8506±0.0020 | 0.8889±0.0020 |
| | rht | 28.24 s | 0.7182±0.0094 | 0.8000±0.0040 | 0.8552±0.0048 | 0.8959±0.0044 |
| | **Δ** | **1.78×** | +0.0128 | +0.0167 | +0.0047 | +0.0070 |

Pooled Δ across all 12 cells: **+0.0034**, against a per-cell seed spread of
±0.002–0.013. Statistically indistinguishable, with the sign slightly
favouring RHT.

Worth noting for calibration: these real corpora have a max-coordinate-variance
ratio of only 2.0–2.6×, far milder than the synthetic anisotropic case. Even
single-round RHT scores fine on them. The failure in step 3 is real but needs
stronger anisotropy than these three corpora carry — which is precisely why
the theory steps had to come first. Ranking the recall table above the
independence measurement would have shipped the bug.

## Two optimizations measured and rejected

**Operator-form FFHT (applying the transform per batch instead of
materializing a dense matrix).** The obvious next step, and it loses badly:

| d | k | encode 10k rows, dense BLAS | operator FWHT | rotation memory |
|---|---|---|---|---|
| 768 | 8 | 0.74 s | 9.18 s (**0.08×**) | 37.7 MB → 0.098 MB |
| 1024 | 8 | 0.79 s | 14.40 s (**0.05×**) | 67.1 MB → 0.131 MB |
| 3072 | 8 | 11.68 s | 79.39 s (**0.15×**) | 604 MB → 0.393 MB |

7–20× *slower* despite ~100× fewer flops: NumPy's FWHT is memory-bound and
pays a fancy-index permutation copy per round, while OpenBLAS `sgemm` runs
near peak. Build time drops to microseconds and rotation memory by ~1000×,
but neither is worth a 10× encode regression. A C/SIMD FFHT kernel would
change this arithmetic; NumPy alone does not. Hence `rht_rotation` returns a
dense matrix.

**Thread-parallel Haar construction across the `k` stacks.** `np.linalg.qr`
releases the GIL, so this looked free. It is 2.6–4.0× *slower* at every `d`
tested — OpenBLAS already parallelizes `dgeqrf` across all cores, and `k`
concurrent factorizations just oversubscribe them.

## What changed in the library

- `remax.rotation.rht_rotation` — dense RHT for any even `d`, `rounds`
  floored at 2 and values below that rejected with a pointer to this document.
- `rotation="haar"|"rht"` on `SignBitQuantizer` and
  `StackedSignBitQuantizer`; `haar` remains the default.
- `StackedSignBitQuantizer.rotations_` is now a **zero-copy view** onto the
  `(d, k·d)` projection matrix rather than a second copy of the stack. This
  is independent of the rotation question and applies to every user: rotation
  memory halves (604 MB → 302 MB at `d=3072, k=8`), the transient `(k, d, d)`
  build buffer is gone, and the two representations can no longer desync.

## Not done here

**`Corpus` cannot select a rotation.** Its on-disk header (`_MAGIC`, v1, 32
bytes) records `(n, d, seed)` and nothing else, and `Corpus.load` reconstructs
the quantizer from those three. Passing `rotation="rht"` through `build`
would write an index that `load` silently reopens under Haar — codes that
still decode, just from the wrong rotation, detectable only as degraded
recall. Header bytes 6–7 are reserved and could carry a rotation tag, but
spending them is a format-version decision, so it is left to a deliberate
change rather than smuggled in here.

**The remex-vs-remax scorer bias** raised in the back half of #59 is
untouched. Nothing in this repo references `score_fidelity.py` or the
`jina-remex-vs-remax` experiment, so there is no remax-side artifact to
correct; the re-check belongs in `oaustegard/experiments`.
