## remax v0.0.0 baseline — R@10 vs float32

- **Library version**: remax v0.0.0
- **Eval metric**: R@10 vs float32 inner-product ground truth (computed on raw, un-centered vectors).
- **Protocol**: 100 held-out queries per dataset, corpus = remainder. Query split seed = 99, quantizer seed = 42.
- **Centering**: corpus and queries are centered by the corpus mean before encoding. Pure SimHash assumes mean-zero data; real embeddings (SPECTER2 has one dim with mean ≈ 15.5) violate this. Lloyd-Max 1-bit boundaries are adaptive per dimension, so they implicitly center; the SimHash-equivalent is `sign(X - corpus.mean(0))`. Disable with `--no-center`.
- **Hardware**: pure NumPy on CPU. No SIMD/Numba/GPU paths (those are post-v0.1.0 by design — see CLAUDE.md anti-goals).

| dataset       | n      | d   | 1-bit | k=2   | k=4   | k=8   |
|---------------|--------|-----|-------|-------|-------|-------|
| SPECTER2      |   9900 | 768 | 0.635 | 0.676 | 0.706 | 0.718 |
| MiniLM-L6-v2  | — | — | — | — | — | — |
| GloVe-300d    | — | — | — | — | — | — |

