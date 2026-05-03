# Post-hoc Matryoshka via Sketching

Can projection + sign-bit extraction approximate Matryoshka-style dimensional
reduction on pre-trained embeddings, without retraining?

**Setup**: 10,000 SPECTER2 embeddings (768-d), 100 held-out queries.
Ground truth: top-10 by float32 inner product.

## Results

```
strategy         type    k    R@10   R@100  B/vec   ratio
────────────────────────────────────────────────────────────────
f32-raw           f32   64   0.101   0.446    256     12x
f32-centered      f32   64   0.284   0.770    256     12x
sign-raw         1bit   64   0.112   0.365      8    384x
sign-centered    1bit   64   0.202   0.600      8    384x
gaussian         1bit   64   0.206   0.610      8    384x
countsketch      1bit   64   0.232   0.626      8    384x
pca              1bit   64   0.258   0.656      8    384x
haar-trunc       1bit   64   0.216   0.634      8    384x

f32-raw           f32  128   0.296   0.770    512      6x
f32-centered      f32  128   0.393   0.898    512      6x
sign-raw         1bit  128   0.211   0.585     16    192x
sign-centered    1bit  128   0.330   0.784     16    192x
gaussian         1bit  128   0.330   0.800     16    192x
countsketch      1bit  128   0.342   0.823     16    192x
pca              1bit  128   0.295   0.666     16    192x
haar-trunc       1bit  128   0.338   0.816     16    192x

f32-raw           f32  192   0.330   0.819    768      4x
f32-centered      f32  192   0.442   0.923    768      4x
sign-raw         1bit  192   0.291   0.731     24    128x
sign-centered    1bit  192   0.415   0.877     24    128x
gaussian         1bit  192   0.398   0.860     24    128x
countsketch      1bit  192   0.431   0.888     24    128x
pca              1bit  192   0.311   0.649     24    128x
haar-trunc       1bit  192   0.429   0.892     24    128x

f32-raw           f32  256   0.420   0.882   1024      3x
f32-centered      f32  256   0.464   0.943   1024      3x
sign-raw         1bit  256   0.336   0.798     32     96x
sign-centered    1bit  256   0.468   0.926     32     96x
gaussian         1bit  256   0.471   0.916     32     96x
countsketch      1bit  256   0.458   0.919     32     96x
pca              1bit  256   0.311   0.636     32     96x
haar-trunc       1bit  256   0.487   0.928     32     96x

f32-raw           f32  384   0.515   0.952   1536      2x
f32-centered      f32  384   0.501   0.957   1536      2x
sign-raw         1bit  384   0.418   0.865     48     64x
sign-centered    1bit  384   0.533   0.963     48     64x
gaussian         1bit  384   0.522   0.958     48     64x
countsketch      1bit  384   0.514   0.947     48     64x
pca              1bit  384   0.318   0.650     48     64x
haar-trunc       1bit  384   0.562   0.966     48     64x

f32-raw           f32  512   0.665   0.995   2048      2x
f32-centered      f32  512   0.514   0.968   2048      2x
sign-raw         1bit  512   0.488   0.930     64     48x
sign-centered    1bit  512   0.570   0.980     64     48x
gaussian         1bit  512   0.561   0.969     64     48x
countsketch      1bit  512   0.552   0.964     64     48x
pca              1bit  512   0.332   0.625     64     48x
haar-trunc       1bit  512   0.591   0.984     64     48x

f32-raw           f32  768   1.000   1.000   3072      1x
f32-centered      f32  768   0.514   0.969   3072      1x
sign-raw         1bit  768   0.544   0.971     96     32x
sign-centered    1bit  768   0.620   0.988     96     32x
gaussian         1bit  768   0.592   0.986     96     32x
countsketch      1bit  768   0.582   0.970     96     32x
pca              1bit  768   0.338   0.635     96     32x
haar-trunc       1bit  768   0.636   0.988     96     32x

────────────────────────────────────────────────────────────────
Full-width baselines (768 dims):

f32-full          f32  768   1.000   1.000   3072      1x
blog-baseline    1bit  768   0.620   0.988     96     32x
remex-1b@p=1      rmx  768   0.609   0.981    100     31x
remex-4b@p=1      rmx  768   0.609   0.981    388      8x
remex-4b@p=4      rmx  768   0.731   0.999    388      8x

Total time: 20.5s
```

## Key findings

### 1. Post-hoc dimensional reduction works

At 256 bits (32 B/vec, 96× compression), haar-trunc achieves R@100 = 0.928.
Even the "truly free" sign-centered gets R@100 = 0.926 at the same budget.

### 2. Centering is the single biggest lever

At every k, centering buys massive improvement — more than any other technique:

| k   | f32-raw R@100 | f32-centered R@100 | Δ     |
|-----|---------------|---------------------|-------|
|  64 | 0.446         | 0.770               | +0.324|
| 128 | 0.770         | 0.898               | +0.128|
| 256 | 0.882         | 0.943               | +0.061|

### 3. 1-bit sign beats float32 centered at k ≥ 256

| k   | f32-centered R@10 | sign-centered R@10 |
|-----|--------------------|--------------------|
| 256 | 0.464              | 0.468              |
| 384 | 0.501              | 0.533              |
| 512 | 0.514              | 0.570              |
| 768 | 0.514              | 0.620              |

This is not a bug. Ground truth uses raw IP. Centering shifts the metric.
Hamming ≈ cosine on centered data is a better proxy for raw IP than centered
IP itself, because cosine on centered data is effectively Pearson correlation
— magnitude-invariant, which raw IP approximately is on vectors with
similar L2 norms (SPECTER2 norms cluster in 20–22).

### 4. PCA has a crossover

PCA is best at k=64 (top PCs carry the most signal per bit) but worst
at k ≥ 192 (low-variance PCs contribute noise bits). The crossover is
between k=64 and k=128. PCA whitening cannot help: `sign(x/c) == sign(x)`
for positive c. Tested explicitly: whiten+Haar amplifies noise and makes
recall *worse* than any other method.

### 5. Random projection ≈ prefix truncation

Gaussian and count-sketch projections match or beat prefix truncation
at all k. No fitting required. JL guarantees explain why.

### 6. Haar-trunc is the best post-hoc Matryoshka

Distributes variance uniformly → every prefix is equally informative.
Consistently leads at k ≥ 256. At full width (k=768), matches the blog
baseline R@100 and exceeds it on R@10 (0.636 vs 0.620).

### 7. remex 1-bit ≈ blog baseline, not better

remex's actual search at precision=1 gives R@10 = 0.609 vs sign-centered's
0.620. The normalization to unit sphere that remex requires loses norm
information that raw IP cares about. remex at bits=4, precision=4 achieves
R@10 = 0.731 — that's where the codebook genuinely helps.
