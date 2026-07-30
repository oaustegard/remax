# LFM2.5-Embedding-350M under vector quantization — BEIR SciFact

**Question:** LFM2.5's embeddings ship with no Matryoshka support and no published
quantization guidance. Does the model lend itself to vector quantization anyway?

**Answer: yes, unusually well.** 1-bit codes at 128 B/vector (32x smaller) retain
98.5% of fp32 nDCG@10 standalone, and a top-25 fp32 rescore of that same 128-byte
index recovers fp32 quality outright. The missing Matryoshka support turns out to
cost nothing that matters, because quantizing all 1024 dims dominates truncating
to fewer — by a wide margin, at a fraction of the bytes.

Corpus: BEIR SciFact, 5183 docs / 300 test queries / 339 judgments. Encoder run at
fp32 on CPU, CLS pooling, `query: ` / `document: ` prefixes, L2-normalized. Single
seed (42). Reproduce with `bench/embed_lfm25.py` then `bench/eval_lfm25.py` and
`bench/rerank_lfm25.py`.

## Baseline sanity

fp32 nDCG@10 = **0.7122**. LiquidAI's own MTEB entry reports NanoSciFact 0.7507,
and NanoBEIR is known to read high against full BEIR; competent small dense
retrievers cluster at 0.71-0.77 on full SciFact (e5-base-v2 0.7194, jina-v3
0.7231, bge-base 0.7434). 0.7122 lands where it should, which is the evidence that
the prompt prefixes and CLS pooling were applied correctly — Liquid warns that
omitting the prefixes silently degrades retrieval, and that failure mode would have
shown up here as a number in the 0.5s.

## The prediction, registered before any retrieval ran

`bench/geometry.py` measures the embedding geometry and emits a falsifiable call on
which codec family should win. On this corpus:

| metric | value | isotropic reference |
|---|--:|--:|
| post-rotation sigma ratio | **0.9370** | 1.0000 |
| post-rotation kurtosis | -0.010 | 0.000 |
| KS reject fraction | 0.00 | 0.00 |
| participation ratio | 118.3 / 1024 | 1024 |
| top-10 eigenvalue fraction | 0.2068 | 0.0098 |
| off-center ratio | 0.3446 | 0 |

Regime: **isotropic**. Prediction: *"remex dominates at every matched byte budget;
4-bit near-lossless."*

**The prediction held at every matched byte budget.** remex beat remax at 128 B
(0.7018 vs 0.6772), 256 B (0.7042 vs 0.6989), and 512 B (0.7110 vs 0.7078), and
4-bit came in at 99.8% of fp32. This is the opposite of the SPECTER2 result already
in this repo, where 1-bit beat 2- and 3-bit — and it is consistent with the
sigma-ratio being the thing that separates the two cases (0.389 there, 0.937 here).

**Caveat on which diagnostic predicted it.** The participation ratio is 118/1024 —
only 11.6% effective dimensionality, which reads as strongly anisotropic. The
sigma ratio says near-isotropic. They disagree, and *the sigma ratio is the one
that was right*. The Haar rotation isotropizes the marginals even when the spectrum
is concentrated, which is exactly what Lloyd-Max needs. Do not use the eigenvalue
spectrum to predict codec ordering.

## Matched byte budgets

| B/vec | ratio | winner | nDCG@10 | % of fp32 |
|--:|--:|---|--:|--:|
| 4096 | 1x | fp32 d=1024 | 0.7122 | 100.0% |
| 1024 | 4x | int8 d=1024 | 0.7122 | 100.0% |
| 512 | 8x | remex 4bit | 0.7110 | 99.8% |
| 384 | 10.7x | remex 3bit | 0.7097 | 99.6% |
| 256 | 16x | remex 2bit | 0.7042 | 98.9% |
| 128 | **32x** | **remex 1bit** | **0.7018** | **98.5%** |
| 64 | 64x | remax 1bit d=512 | 0.6501 | 91.3% |
| 32 | 128x | remax 1bit d=256 | 0.5722 | 80.3% |

int8 is lossless to four decimal places at 4x. The interesting regime starts below
that, and 128 B/vec is where the curve is still nearly flat — 32x compression for
1.5% nDCG.

## Quantize, don't truncate — the answer to "no Matryoshka"

This is the practically decisive comparison, because it is the one the missing MRL
support appears to force on you, and it turns out not to:

| approach | B/vec | nDCG@10 |
|---|--:|--:|
| fp32 truncated to 256 dims | 1024 | 0.6572 |
| **remex 1-bit over all 1024 dims** | **128** | **0.7018** |

**8x less storage and 6.8% better quality.** Prefix truncation is the wrong lever
on this model; bit-depth is the right one.

The truncation curve does confirm the cost of not training for MRL. Against
jina-v3, which is MRL-trained, on the same task:

| dims | LFM2.5 | jina-v3 (MRL) |
|--:|--:|--:|
| 1024 | 0.7122 | 0.7231 |
| 512 | 0.7042 (-1.1%) | 0.7247 (+0.2%) |
| 256 | 0.6572 (-7.7%) | 0.7152 (-1.1%) |
| 128 | 0.6085 (-14.6%) | 0.6995 (-3.3%) |
| 64 | 0.5092 (-28.5%) | 0.6494 (-10.2%) |

LFM2.5 degrades 3-7x faster under truncation depending on depth (7.1x at 256 dims,
4.5x at 128, 2.8x at 64; at 512 dims jina-v3 actually gains, so the ratio there is
not meaningful). So the "no Matryoshka" caveat is real — it just doesn't bind,
because quantization is strictly the better axis. (jina-v3 figures are published,
measured on a different run; treat the comparison as directional.)

## Two-stage: 32x compression, no measured loss

Shortlist from the compressed index, reorder the shortlist by exact fp32 score.

| stage-1 codec | B/vec | stage-1 alone | +rescore top-10 | +top-25 | +top-50 |
|---|--:|--:|--:|--:|--:|
| remax 1bit | 128 | 0.6772 | 0.7044 | **0.7131** | 0.7137 |
| remex 1bit | 128 | 0.7018 | 0.7132 | 0.7120 | 0.7122 |
| remax k=2 | 256 | 0.6989 | 0.7144 | 0.7134 | 0.7133 |
| remex 4bit | 512 | 0.7110 | 0.7122 | 0.7122 | 0.7122 |

A **top-25 rescore over a 128-byte index fully recovers fp32 nDCG@10** — a 25-row
shortlist out of 5183. That is far cheaper than the 3-4x oversampling the published
conventions (Elasticsearch BBQ, Qdrant) recommend for binary quantization.

Values marginally above 1.0000 of baseline (+0.001 to +0.002) are noise on a
300-query set, not real gains — rescoring a shortlist can reorder ties favorably.
Read them as "recovered", not "improved".

## Negative / surprising results

**Centering did not help.** The geometry flagged an off-center ratio of 0.3446 and
predicted centering would matter. It didn't: uncentered 1-bit scored 0.6808 vs
0.6772 centered — marginally *better*, and within noise either way. The
`centering_matters` heuristic in `geometry.py` is not validated by this run and
should not be trusted as-is. (Centering remains load-bearing on GEMINI per
SKETCH_MATRYOSHKA_GEMINI.md, so this is embedder-specific, not a general refutation.)

**remax loses cleanly here.** At every matched budget where both compete, remex
wins. This repo's 1-bit codes are the right tool on anisotropic encoders like
SPECTER2; on LFM2.5 they are not. Same code, opposite curve — which is the standing
conclusion of `docs/research/matryoshka-and-quantization.md` and now has a second
confirming data point.

> **Superseded, 2026-07-30.** The paragraph above attributed the remax/remex gap
> to Lloyd-Max vs SimHash. That was wrong. The gap is **symmetric vs asymmetric
> scoring**: remex decodes to float and takes an inner product, while remax's
> `search` binarizes the query too. Giving remax an asymmetric path
> (`search_asymmetric`, float query against the same sign-bit index) closes it
> almost exactly — 0.7020 vs remex's 0.7018 at an identical 128 B/vec. See
> `LFM25_ASYMMETRIC.md`. The codec comparison at matched bytes stands; the
> explanation for it does not.

## Asymmetric scoring

Prompted by Exa's web-scale index, which stores document sign bits but keeps the
query in float. The query is one vector per search and occupies no index storage,
so binarizing it discards precision for nothing.

| dim | B/vec | symmetric | asymmetric | delta |
|--:|--:|--:|--:|--:|
| 1024 | 128 | 0.6772 | **0.7020** | +0.0248 |
| 512 | 64 | 0.6501 | 0.6859 | +0.0358 |
| 256 | 32 | 0.5722 | 0.6191 | +0.0469 |
| 128 | 16 | 0.4214 | **0.5282** | +0.1067 |

Free at the index level, and the harder the compression the more the query
precision carries. It does **not** reach fp32 (0.7122) standalone — it closes the
gap to remex, not to uncompressed. Reproduce with `bench/asymmetric_lfm25.py`.

### Asymmetry substitutes for stacking

Stacking and asymmetry buy down the same error — the variance of a similarity
estimated from sign bits. Stacking pays index bytes for it; asymmetry pays nothing.

| codec | B/vec | symmetric | asymmetric | delta |
|---|--:|--:|--:|--:|
| k=1 | 128 | 0.6772 | **0.7020** | +0.0248 |
| k=2 | 256 | 0.6989 | 0.7022 | +0.0033 |
| k=4 | 512 | 0.7078 | 0.7111 | +0.0033 |

Two consequences worth acting on:

- **Asymmetric k=1 (128 B, 0.7020) beats symmetric k=2 (256 B, 0.6989).** Turning
  on asymmetry is worth more than doubling the index.
- **Under asymmetric scoring, k=2 buys essentially nothing** (0.7022 vs 0.7020 for
  k=1 at twice the bytes). The k=2 stack exists to reduce estimator variance, and
  asymmetry has already taken that slack. The gain collapsing from +0.0248 to
  +0.0033 the moment you stack is the same fact seen from the other side.

If you are scoring asymmetrically, spend bytes on k=4 or not at all — k=2 is a
dominated configuration on this embedder.

## Scope and limits

- One dataset (SciFact), one seed, 300 queries. Differences under ~0.005 nDCG are
  not resolvable here. The bit-depth *ordering* is robust; the exact percentages
  are not.
- SciFact is English-only and scientific. LFM2.5 is a 15-language model; multilingual
  and out-of-domain behaviour under quantization is untested.
- Documents were truncated at the model's 512-token `max_seq_length`; SciFact
  abstracts average ~1500 chars so this affects a minority of rows.
- Encoder run at fp32. Model-weight quantization (the GGUF/ONNX int8 builds) is a
  separate axis and composes independently — see NEMOTRON_MASTER.md, where NVFP4
  weight quantization cost ~0 and stacked with vector quantization.
- No published vector-quantization numbers exist for any LFM2.5 model, so there is
  nothing external to check these against. They are the first of their kind that we
  could find.
