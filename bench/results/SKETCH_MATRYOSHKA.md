# Post-hoc Matryoshka via Sketching

Can random projection + sign-bit extraction approximate Matryoshka-style
dimensional reduction on pre-trained embeddings?

**Setup**: 10,000 SPECTER2 embeddings (768-d), 100 held-out queries.
Ground truth: top-10 by float32 inner product.

**TL;DR**: Yes. At 256 bits (32 bytes/vec, 96× compression), `haar-trunc`
achieves R@100 = 0.928. Random projection (gaussian, countsketch) is
competitive with prefix truncation and requires no fitting. PCA is the
worst basis for sign-bit extraction due to variance concentration.

```
strategy            k  bits    R@10   R@100  B/vec   ratio
──────────────────────────────────────────────────────────────
prefix             64    64   0.202   0.600      8    384x
gaussian           64    64   0.213   0.602      8    384x
countsketch        64    64   0.235   0.636      8    384x
pca                64    64   0.258   0.656      8    384x
haar-trunc         64    64   0.216   0.634      8    384x

prefix            128   128   0.330   0.784     16    192x
gaussian          128   128   0.326   0.799     16    192x
countsketch       128   128   0.332   0.809     16    192x
pca               128   128   0.295   0.666     16    192x
haar-trunc        128   128   0.338   0.816     16    192x

prefix            192   192   0.415   0.877     24    128x
gaussian          192   192   0.393   0.898     24    128x
countsketch       192   192   0.428   0.890     24    128x
pca               192   192   0.311   0.649     24    128x
haar-trunc        192   192   0.429   0.892     24    128x

prefix            256   256   0.468   0.926     32     96x
gaussian          256   256   0.474   0.913     32     96x
countsketch       256   256   0.465   0.921     32     96x
pca               256   256   0.311   0.636     32     96x
haar-trunc        256   256   0.487   0.928     32     96x

prefix            384   384   0.533   0.963     48     64x
gaussian          384   384   0.542   0.956     48     64x
countsketch       384   384   0.501   0.957     48     64x
pca               384   384   0.318   0.650     48     64x
haar-trunc        384   384   0.562   0.966     48     64x

prefix            512   512   0.570   0.980     64     48x
gaussian          512   512   0.557   0.963     64     48x
countsketch       512   512   0.536   0.967     64     48x
pca               512   512   0.332   0.625     64     48x
haar-trunc        512   512   0.591   0.984     64     48x

prefix            768   768   0.620   0.988     96     32x
gaussian          768   768   0.594   0.984     96     32x
countsketch       768   768   0.562   0.971     96     32x
pca               768   768   0.338   0.635     96     32x
haar-trunc        768   768   0.636   0.988     96     32x

──────────────────────────────────────────────────────────────
blog-baseline     768   768   0.620   0.988     96     32x

Total time: 8.2s
```

## Key findings

1. **Post-hoc dimensional reduction works.** Haar-trunc at 256 bits
   gives R@100 = 0.928 — useful for a two-stage retriever's Stage 1.

2. **Random projection ≈ prefix truncation.** Gaussian and count-sketch
   projections are competitive with naive prefix truncation at all bit
   budgets. No fitting required — pure post-hoc.

3. **PCA is the worst basis.** It concentrates variance in the top
   components, but sign bits weight all dimensions equally. At k=768,
   the ~500 low-variance bits dominate Hamming distance with noise.

4. **PCA whitening can't help.** `sign(x/c) == sign(x)` for positive c.
   Scaling before signing is a no-op. Whitening then re-rotating
   (whiten+Haar) amplifies noise dimensions, making recall *worse*.

5. **Signal uniformity > variance uniformity.** Modern encoders
   distribute signal roughly uniformly across dimensions. That's why
   naive centered sign extraction works. Haar rotation slightly
   improves uniformity; PCA destroys it.

6. **Haar-trunc is the best post-hoc Matryoshka analog** — it
   distributes variance uniformly across all dimensions, making any
   prefix equally informative. At k≥256, it consistently leads.
