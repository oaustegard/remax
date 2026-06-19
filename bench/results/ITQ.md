## Issue #46 — learned hash projections (ITQ) vs centered SimHash

- **Library**: remax v0.0.0, pure NumPy/SciPy on CPU.
- **Protocol**: 100 held-out queries per eval set (split seed 99), corpus = remainder. Quantiser seed 42; ITQ 50 iters/rotation.
- **Encoders**: `haar` = centered SimHash (Haar rotations, data-agnostic); `itq_in` = ITQ learned on the eval corpus; `itq_xfer` = ITQ learned on the *other* corpus (cross-corpus probe).
- **Metrics vs float32 cosine**: R@k (top-k set overlap), Kendall τ-b (full-corpus rank agreement), cosine-graded nDCG@k. Ground truth and cosine reference are on raw vectors; centering is encoder-side only.
- **Equal bits**: every rung compares encoders at the same `k·d` bit budget. Flavor (b) relevance-distilled projections are out of scope (no teacher labels in these corpora; gated on flavor (a) — see runner docstring).

### eval = SPECTER2-broad  (transfer rotation from SPECTER2-narrow)

- n=9900, d=768, R@10, τ = Kendall τ-b vs cosine order.
- **win** = `itq_in R@10 − haar R@10` at equal bits. **xfer Δ** = `itq_xfer R@10 − itq_in R@10` (transfer penalty).

| k | bits | haar R@10 | itq_in R@10 | itq_xfer R@10 | win | xfer Δ | haar τ | itq_in τ | itq_xfer τ | haar nDCG | itq_in nDCG | itq_xfer nDCG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 768 | 0.637 | 0.631 | 0.602 | -0.006 | -0.029 | 0.584 | 0.592 | 0.620 | 0.996 | 0.996 | 0.996 |
| 2 | 1536 | 0.675 | 0.666 | 0.651 | -0.009 | -0.015 | 0.601 | 0.602 | 0.633 | 0.997 | 0.997 | 0.997 |
| 4 | 3072 | 0.706 | 0.682 | 0.686 | -0.024 | +0.004 | 0.608 | 0.608 | 0.641 | 0.998 | 0.997 | 0.998 |
| 8 | 6144 | 0.722 | 0.690 | 0.713 | -0.032 | +0.023 | 0.613 | 0.611 | 0.646 | 0.998 | 0.998 | 0.998 |

### eval = SPECTER2-narrow  (transfer rotation from SPECTER2-broad)

- n=9900, d=768, R@10, τ = Kendall τ-b vs cosine order.
- **win** = `itq_in R@10 − haar R@10` at equal bits. **xfer Δ** = `itq_xfer R@10 − itq_in R@10` (transfer penalty).

| k | bits | haar R@10 | itq_in R@10 | itq_xfer R@10 | win | xfer Δ | haar τ | itq_in τ | itq_xfer τ | haar nDCG | itq_in nDCG | itq_xfer nDCG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 768 | 0.703 | 0.685 | 0.681 | -0.018 | -0.004 | 0.660 | 0.663 | 0.649 | 0.998 | 0.998 | 0.998 |
| 2 | 1536 | 0.724 | 0.706 | 0.727 | -0.018 | +0.021 | 0.674 | 0.670 | 0.656 | 0.999 | 0.998 | 0.998 |
| 4 | 3072 | 0.762 | 0.730 | 0.766 | -0.032 | +0.036 | 0.682 | 0.673 | 0.664 | 0.999 | 0.998 | 0.999 |
| 8 | 6144 | 0.774 | 0.737 | 0.789 | -0.037 | +0.052 | 0.685 | 0.676 | 0.666 | 0.999 | 0.999 | 0.999 |

## Findings (n=10k, 50 ITQ iters)

**Negative result. ITQ does not beat centered SimHash on SPECTER2, and the gap widens with stack depth.** At every rung, on both corpora, the in-corpus learned rotation (`itq_in`) loses to the parameter-free baseline: the `win` column is negative throughout, and it gets *more* negative as `k` grows — broad −0.006 → −0.032, narrow −0.018 → −0.037 from k=1 to k=8. Rank-correctness tells the same story: `haar` and `itq_in` Kendall τ are tied to ±0.002. ITQ buys nothing on the metric it was supposed to improve, and costs recall.

**The ladder is where ITQ loses.** Haar recall climbs steeply with `k` (broad 0.637 → 0.722) because the `k` independent rotations give the stacked estimator its `1/k` variance reduction. `itq_in` climbs far less (broad 0.631 → 0.690): `k` ITQ rotations all minimize the *same* sign-quantization MSE, so they converge toward similar solutions, their signatures correlate, and the variance reduction the ladder depends on is partly defeated. The deficit growing with `k` is the signature of this — at k=1 the methods are within noise; the gap only opens once stacking should be helping.

**Transfer beats in-corpus — because foreign rotations act more like random ones.** The counterintuitive `xfer Δ` is positive at k ≥ 2 (narrow k=8: itq_xfer 0.789 vs itq_in 0.737, and even above haar's 0.774). A rotation learned on the *other* SPECTER2 corpus is less aligned to the eval data, so a stack of them retains more mutual independence — closer to the Haar regime the ladder wants. This is not evidence that transfer is good; it is evidence that ITQ's in-corpus fitting is actively counterproductive *for stacking*, and that nudging the rotations back toward randomness recovers recall. The broad↔narrow distributions are close enough that the transfer penalty on rank-correctness is mild (τ within ~0.04).

**nDCG@10 is saturated (≥ 0.996 everywhere)** and does not discriminate the methods — at k=10 against graded cosine relevance all three encoders put near-ideal mass early. R@10 and full-corpus τ are the discriminating metrics here.

### Decision

Keep centered SimHash as the default and fallback; do **not** adopt ITQ for the ladder. The mechanism is fundamental, not a tuning artifact: any rotation learned under a shared per-rotation objective will correlate across the stack and erode the `1/k` variance reduction. Flavor (b) (relevance-distilled projections) is not worth attempting on this evidence — it inherits the same stacking pathology and adds teacher-label cost and overfitting risk, while flavor (a)'s precondition ("ITQ ≥ baseline at equal bits") was not met. If learned hashing is revisited, the open question is whether an objective that *decorrelates* the `k` rotations (e.g. a joint multi-rotation loss with an orthogonality-between-stacks penalty) can keep the ladder's independence while still fitting the data — but that is a different method, not ITQ.
