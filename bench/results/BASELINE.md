## remax v0.0.0 baseline — R@10 vs float32

- **Library version**: remax v0.0.0
- **Eval metric**: R@10 vs float32 inner-product ground truth
- **Protocol**: 100 held-out queries per dataset, corpus = remainder. Query split seed = 99, quantizer seed = 42.
- **Hardware**: pure NumPy on CPU. No SIMD/Numba/GPU paths (those are post-v0.1.0 by design — see CLAUDE.md anti-goals).

| dataset       | n      | d   | 1-bit | k=2   | k=4   | k=8   |
|---------------|--------|-----|-------|-------|-------|-------|
| SPECTER2      |   9900 | 768 | 0.468 | 0.547 | 0.621 | 0.676 |
| MiniLM-L6-v2  | — | — | — | — | — | — |
| GloVe-300d    | — | — | — | — | — | — |

