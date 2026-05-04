# Post-hoc Matryoshka via Sketching — Gemini

Repeats the [SPECTER2 sketch_matryoshka experiment](SKETCH_MATRYOSHKA.md) on
Gemini `gemini-embedding-001` embeddings. Same 10K NLP-broad papers, same
strategies, same ground truth. Tests whether Matryoshka training and L2
normalization change the post-hoc compression story.

**Setup**: 10,000 Gemini embeddings (3072-d, Matryoshka, L2-normalized, norms = 1.000),
100 held-out queries. Ground truth: top-10 by float32 inner product on full 3072-d.

## Results

```
strategy         type    k    R@10   R@100  B/vec   ratio
────────────────────────────────────────────────────────────────
f32-raw           f32   64   0.018   0.149    256     48x
f32-centered      f32   64   0.226   0.630    256     48x
sign-raw         1bit   64   0.065   0.265      8   1536x
sign-centered    1bit   64   0.140   0.435      8   1536x
gaussian         1bit   64   0.144   0.445      8   1536x
countsketch      1bit   64   0.134   0.460      8   1536x
pca              1bit   64   0.263   0.684      8   1536x
haar-trunc       1bit   64   0.120   0.420      8   1536x

f32-raw           f32  128   0.078   0.308    512     24x
f32-centered      f32  128   0.347   0.819    512     24x
sign-raw         1bit  128   0.166   0.508     16    768x
sign-centered    1bit  128   0.266   0.676     16    768x
gaussian         1bit  128   0.267   0.666     16    768x
countsketch      1bit  128   0.286   0.707     16    768x
pca              1bit  128   0.360   0.756     16    768x
haar-trunc       1bit  128   0.258   0.693     16    768x

f32-raw           f32  256   0.347   0.781   1024     12x
f32-centered      f32  256   0.484   0.933   1024     12x
sign-raw         1bit  256   0.338   0.761     32    384x
sign-centered    1bit  256   0.432   0.873     32    384x
gaussian         1bit  256   0.419   0.857     32    384x
countsketch      1bit  256   0.405   0.860     32    384x
pca              1bit  256   0.382   0.753     32    384x
haar-trunc       1bit  256   0.415   0.857     32    384x

f32-raw           f32  384   0.543   0.950   1536      8x
f32-centered      f32  384   0.540   0.953   1536      8x
sign-raw         1bit  384   0.434   0.876     48    256x
sign-centered    1bit  384   0.520   0.944     48    256x
gaussian         1bit  384   0.496   0.910     48    256x
countsketch      1bit  384   0.455   0.911     48    256x
pca              1bit  384   0.373   0.737     48    256x
haar-trunc       1bit  384   0.469   0.929     48    256x

f32-raw           f32  512   0.657   0.990   2048      6x
f32-centered      f32  512   0.547   0.969   2048      6x
sign-raw         1bit  512   0.506   0.929     64    192x
sign-centered    1bit  512   0.574   0.963     64    192x
gaussian         1bit  512   0.533   0.940     64    192x
countsketch      1bit  512   0.504   0.952     64    192x
pca              1bit  512   0.389   0.713     64    192x
haar-trunc       1bit  512   0.529   0.947     64    192x

f32-raw           f32  768   0.874   1.000   3072      4x
f32-centered      f32  768   0.575   0.978   3072      4x
sign-raw         1bit  768   0.608   0.964     96    128x
sign-centered    1bit  768   0.624   0.977     96    128x
gaussian         1bit  768   0.590   0.969     96    128x
countsketch      1bit  768   0.568   0.980     96    128x
pca              1bit  768   0.359   0.657     96    128x
haar-trunc       1bit  768   0.589   0.972     96    128x

f32-raw           f32 1024   0.870   1.000   4096      3x
f32-centered      f32 1024   0.571   0.978   4096      3x
sign-raw         1bit 1024   0.650   0.980    128     96x
sign-centered    1bit 1024   0.653   0.986    128     96x
gaussian         1bit 1024   0.601   0.979    128     96x
countsketch      1bit 1024   0.633   0.984    128     96x
pca              1bit 1024   0.337   0.641    128     96x
haar-trunc       1bit 1024   0.620   0.980    128     96x

f32-raw           f32 1536   0.938   1.000   6144      2x
f32-centered      f32 1536   0.579   0.979   6144      2x
sign-raw         1bit 1536   0.694   0.990    192     64x
sign-centered    1bit 1536   0.684   0.991    192     64x
gaussian         1bit 1536   0.654   0.991    192     64x
countsketch      1bit 1536   0.653   0.992    192     64x
pca              1bit 1536   0.315   0.602    192     64x
haar-trunc       1bit 1536   0.659   0.992    192     64x

f32-raw           f32 2048   0.881   1.000   8192      2x
f32-centered      f32 2048   0.572   0.977   8192      2x
sign-raw         1bit 2048   0.719   0.996    256     48x
sign-centered    1bit 2048   0.689   0.994    256     48x
gaussian         1bit 2048   0.671   0.995    256     48x
countsketch      1bit 2048   0.669   0.997    256     48x
pca              1bit 2048   0.316   0.586    256     48x
haar-trunc       1bit 2048   0.686   0.996    256     48x

f32-raw           f32 3072   1.000   1.000  12288      1x
f32-centered      f32 3072   0.580   0.978  12288      1x
sign-raw         1bit 3072   0.764   1.000    384     32x
sign-centered    1bit 3072   0.712   0.995    384     32x
gaussian         1bit 3072   0.687   0.998    384     32x
countsketch      1bit 3072   0.677   0.995    384     32x
pca              1bit 3072   0.319   0.575    384     32x
haar-trunc       1bit 3072   0.708   0.998    384     32x

────────────────────────────────────────────────────────────────
Full-width baselines (3072 dims):

f32-full          f32 3072   1.000   1.000  12288      1x
blog-baseline    1bit 3072   0.712   0.995    384    32x
remex-1b@p=1      rmx 3072   0.786   1.000    388     32x
remex-4b@p=1      rmx 3072   0.786   1.000   1540      8x
remex-4b@p=4      rmx 3072   0.896   1.000   1540      8x

────────────────────────────────────────────────────────────────
Index-selection grid (centered, sign-packed; n_random=5):

select              k    R@10   R@100  B/vec
────────────────────────────────────────────────
prefix            256   0.432   0.873     32
suffix            256   0.418   0.879     32
spaced            256   0.420   0.843     32
random (avg)      256   0.412   0.861     32   ±0.007

prefix            768   0.624   0.977     96
suffix            768   0.588   0.978     96
spaced            768   0.600   0.971     96
random (avg)      768   0.593   0.972     96   ±0.003
```

Run: `bash bench/fetch_gemini_cache.sh && python bench/sketch_matryoshka.py --dataset GEMINI --index-grid` (~98s on a single CPU).

## Key findings

### 1. Matryoshka training does NOT make the prefix uniquely informative for sign-bit retrieval

The hypothesis: Matryoshka training optimizes the prefix, so prefix index
selection should beat random/suffix at sign-packed Hamming search below 768-d.

The result: prefix ≈ suffix ≈ spaced ≈ random at both k=256 and k=768.

| k   | prefix R@100 | suffix R@100 | spaced R@100 | random R@100 (±std) |
|-----|--------------|--------------|--------------|---------------------|
| 256 | 0.873        | 0.879        | 0.843        | 0.861 ±0.007        |
| 768 | 0.977        | 0.978        | 0.971        | 0.972 ±0.003        |

This is the same pattern SPECTER2 showed (where the encoder was *not*
Matryoshka-trained). Two interpretations:

- Matryoshka loss optimizes float32 IP at trained truncation points, not
  sign-bit Hamming distance. The two metrics decouple — sign-packing
  destroys the magnitude information Matryoshka relied on.
- Below the training floor (Gemini's MRL minimum is 768), no "front-loading"
  is guaranteed. At k=768, prefix and suffix tie at R@100=0.977, meaning
  whatever Matryoshka redistribution exists at the trained boundary is
  invisible after sign-packing.

Either way: **for 1-bit retrieval, you cannot exploit the Matryoshka prefix.**

### 2. L2 normalization removes centering as a lever

SPECTER2 (norms ~20-22): centering was the single biggest improvement at
every k. Gemini (norms = 1.0): centering helps below k=768 but **hurts**
above it.

| k    | Gemini sign-raw R@10 | Gemini sign-centered R@10 |
|------|-----------------------|---------------------------|
|  256 | 0.338                 | 0.432                     |
|  768 | 0.608                 | 0.624                     |
| 1024 | 0.650                 | 0.653                     |
| 1536 | 0.694                 | 0.684                     |
| 3072 | **0.764**             | 0.712                     |

For SPECTER2 the analogous comparison always favors centered. For Gemini,
sign-raw wins at full width. L2-normalized data is already on the unit
sphere; subtracting a small mean and re-binarizing flips bits whose magnitude
sat near zero, adding noise rather than removing bias.

### 3. PCA flips from worst-at-large-k to best-at-tiny-k

For SPECTER2, PCA was worst at k≥192 and best only at k=64. For Gemini, PCA
is worst at k≥384 (R@100 falls to 0.575) but **decisively best at k=64 and
k=128**:

| k   | sign-centered R@100 | pca R@100 |
|-----|---------------------|-----------|
|  64 | 0.435               | **0.684** |
| 128 | 0.676               | **0.756** |
| 256 | **0.873**           | 0.753     |

At extreme compression (k=64, 8 B/vec, 1536× compression), PCA gets recall
that no other strategy reaches. Gemini concentrates more variance in the top
PCs than SPECTER2 does — consistent with a 3072-d encoder trained to be
useful at multiple truncation points.

### 4. haar-trunc is no longer king

For SPECTER2, haar-trunc was the consistent leader at k≥256 because Haar
rotation distributes variance uniformly across all dims, making any prefix
equally good. For Gemini, sign-raw or sign-centered match or beat haar-trunc
at every k≥256, and at k=3072, sign-raw wins outright (R@10 = 0.764 vs
0.708).

The reason: Gemini's native distribution is *already* near-isotropic on the
unit sphere. Haar rotation has nothing to redistribute. Centering+rotation
adds noise rather than signal.

### 5. The 32 B/vec operating point: SPECTER2 still wins per byte

| Encoder  | k   | Best 1-bit R@100 | Strategy        | B/vec |
|----------|-----|-------------------|-----------------|-------|
| SPECTER2 | 256 | **0.928**         | haar-trunc      | 32    |
| Gemini   | 256 | 0.879             | suffix          | 32    |
| Gemini   | 768 | 0.980             | countsketch     | 96    |

At matched 32 bytes, SPECTER2 beats Gemini. To match SPECTER2's R@100=0.928
on Gemini, you need ~64 B/vec (Gemini sign-centered at k=512 gets 0.963).
Matryoshka training does not buy free compression on this benchmark — it
buys you the *ability* to truncate at trained sizes (768/1536/3072) without
catastrophic loss, but it doesn't beat SPECTER2's pre-trained representation
density at small byte budgets.

### 6. Compression arithmetic

| Strategy            | B/vec | Ratio  | R@100 |
|---------------------|-------|--------|-------|
| f32-full (3072)     | 12288 | 1×     | 1.000 |
| sign-raw (3072)     |   384 | 32×    | 1.000 |
| remex-4b@p=4 (3072) |  1540 | 8×     | 1.000 |
| sign-centered (768) |    96 | 128×   | 0.977 |
| sign-centered (256) |    32 | 384×   | 0.873 |
| pca (64)            |     8 | 1536×  | 0.684 |

The 384× compression headline holds — but at lower recall than SPECTER2's
analogous 96× operating point. The wider dynamic range exposes a different
story: at extreme compression (≥1000×), PCA wins; at moderate compression,
sign+centering wins; at low compression, sign-raw wins on L2-normalized data.

## Comparison: SPECTER2 vs Gemini at matched bit budgets

| B/vec | k (S2) | k (Gem) | S2 best R@100 | Gem best R@100 | Winner |
|-------|--------|---------|---------------|-----------------|--------|
|     8 |     64 |      64 | 0.770 (f32-c) | 0.684 (pca)     | S2     |
|    16 |    128 |     128 | 0.898         | 0.756           | S2     |
|    32 |    256 |     256 | 0.928         | 0.879           | S2     |
|    64 |    512 |     512 | 0.984         | 0.963           | S2     |
|    96 |    768 |     768 | 0.988         | 0.980           | S2     |
|   128 |    —   |    1024 | —             | 0.986           | —      |
|   384 |    —   |    3072 | —             | 1.000 (sign-raw)| —      |

SPECTER2 dominates at every matched byte budget where both encoders can be
truncated. The "free" gain from Matryoshka training does not appear in this
benchmark; sign-packing washes it out.

## What this means for remax

- The Matryoshka assumption — that there's a privileged prefix worth
  exploiting in stacked sign-bit signatures — does not hold for 1-bit
  retrieval on Gemini. Treat all dimensions as exchangeable.
- L2 normalization is increasingly common in modern encoders. The
  centering-as-default heuristic from the SPECTER2 work needs a per-encoder
  test: for normalized encoders, sign-raw should be the baseline.
- For very low byte budgets on Matryoshka-trained encoders, PCA is worth
  reaching for — it captures the genuinely concentrated variance that
  Matryoshka loss induces in float32 space, and that survives sign-packing
  better than truncation does.
- haar-trunc is a SPECTER2-specific win. For pre-rotated, normalized
  encoders, no rotation is needed.
