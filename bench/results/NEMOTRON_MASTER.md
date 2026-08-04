# Nemotron-3-Embed-1B: the full compression landscape

One model, every method, matched byte budgets. Answers: full BF16 vs NVIDIA's
NVFP4 vs remax sign codes vs remex Lloyd-Max (2/3/4-bit) vs int8 vs PQ.

Chart: `nemotron_master.png`. Source CSVs: `nemotron_seeds.csv` (remax + float
trunc), `nemotron_remex.csv` (Lloyd-Max), `nemotron_baselines.csv` (int8, PQ),
`nemotron_nvfp4.csv` (NVFP4 + composition). remax/remex rows are mean over
seeds 0–4 (std ≤ 0.01 throughout).

## Two axes of quantization

These are not competitors — they compress different things and **compose**:

- **Model-weight quantization** (NVIDIA's **NVFP4**): shrinks the *model* (GPU
  memory / compute). Output is still a 2048-dim float embedding.
- **Embedding quantization** (remax / remex / int8 / PQ): shrinks the *stored
  vectors* in the index (RAM / disk). Independent of which checkpoint emitted them.

## 1. What NVFP4 (model-weight quant) costs — essentially nothing

Reconstructed on CPU from the published NVFP4 checkpoint (FP4 e2m1 weights ×
FP8 block scales × FP32 global scale; validated at cosine 0.9955 vs the BF16
weights). Encoded embeddings agree with BF16 at **mean cosine 0.9959**.

| model | SciFact nDCG@10 | STS-B ρ | R@10 vs BF16-float |
|-------|-----------------|---------|--------------------|
| BF16 full | 0.8419 | 0.8478 | 1.000 |
| NVFP4 full | 0.8419 | 0.8484 | 0.950 |

Model quantization costs **~0 nDCG / ~0 ρ** on these tasks — it perturbs *which*
documents come back (95% top-10 overlap) but not their relevance quality, since
the swaps are among near-equivalent docs. (Reconstructs NVFP4 weights faithfully;
activation quant not simulated, so a mild upper bound. NVIDIA's own RTEB
BF16→NVFP4 delta is 0.38 nDCG.)

**Composition holds.** remax-1-bit on NVFP4 embeddings ≈ on BF16 (nDCG 0.823 vs
0.820; ρ 0.846 vs 0.845). You can run the smaller NVFP4 model *and* a 32×-smaller
remax index — multiplicative wins on different resources.

## 2. Embedding quant at matched bytes — who owns which budget

SciFact nDCG@10 (retrieval), the discriminating task:

| bytes/vec | remax (sign) | remex (Lloyd-Max) | int8 | PQ | float trunc |
|-----------|:------------:|:-----------------:|:----:|:--:|:-----------:|
| 1024 | 0.838 (stack4) | **0.841** (4-bit) | 0.834 | — | 0.760 |
| 512 | 0.835 (stack2) | **0.839** (2-bit) | 0.817 | — | 0.664 |
| 256 | **0.825** (1-bit) | — | 0.760 | 0.811 | 0.490 |
| 128 | **0.802** (mrl1024) | — | 0.662 | 0.760 | — |
| 64 | **0.737** (mrl512) | — | — | 0.686 | — |
| 32 | 0.627 (mrl256) | — | — | **0.661** | — |

STS-B ρ is forgiving — every method stays ≥ 0.82 down to 256 B; the ordering
matches SciFact but the gaps are smaller.

The frontier splits cleanly by regime:
- **512–1024 B: remex Lloyd-Max leads** (marginally over remax stacked, clearly
  over int8). At 2–4 bits/coord with stored norms, Lloyd-Max reconstructs the
  vector more faithfully than the same bytes spent on independent sign-rotations.
- **64–256 B: remax owns it.** Lloyd-Max at full dimension can't go below 512 B
  (2 bits × 2048). remax's sign codes hit 256 B (1-bit) and its MRL-slice path
  reaches 64–128 B while staying well above PQ and int8-truncation.
- **32 B: PQ edges remax** (0.661 vs 0.627) — the one budget where a trained
  codebook wins, at the cost of a fit step and a training corpus.

## 3. The honest headline: the "broken middle" does NOT reproduce here

remax's origin (the *one-bit-beats-two* blog post) found, on SPECTER2, that
2-bit and 3-bit **Matryoshka-nested** Lloyd-Max codes were rank-broken — 1-bit
beat them. On **Nemotron-3-Embed-1B**, evaluating **independent** Lloyd-Max
quantizers per bit width, that inversion **does not appear**: remex 2/3/4-bit are
monotone and rank-healthy, and at matched bytes they slightly *beat* remax
stacked sign codes (nDCG 0.839 vs 0.835 at 512 B; 0.841 vs 0.838 at 1024 B).

This refines rather than confirms the remax thesis for this model:
- The *nesting* pathology (right-shifting an n-bit code) is a different mechanism
  than the *independent* per-bit quantizers measured here — this experiment does
  not test the nested-extraction path where the blog's inversion lived.
- remax's durable, model-independent advantage is **the aggressive regime**
  (≤ 256 B), which Lloyd-Max cannot reach at full dimension, plus being
  data-oblivious (no codebook, no training corpus, no fit — unlike PQ) and
  scale-free (no stored norms).

## Caveats

BF16 twin stands in for the NVFP4 activation path; two datasets on one model
family (a signal, not a benchmark sweep); SciFact is a 1600-doc subset. remex/PQ
store a float32 norm per vector (+4 B), included in their byte counts.

## Provenance

The ten scripts that produced these CSVs were **deleted on 2026-08-03** — they
benchmarked a third-party embedder, not remax, and were the largest block of
Python in the repository. The CSVs, the PNG and the prose here are the record.
[`NEMOTRON_1BIT.md`](NEMOTRON_1BIT.md#provenance--the-drivers-are-gone-the-results-are-not)
carries the full artifact→script mapping and how to recover a driver from git
history. `bench/fetch_nemotron_cache.sh` is kept: it is the pointer to the
published embeddings the whole family read.
