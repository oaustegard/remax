# Transform selection for a cosine-metric managed ANN index

Driver: `bench/s3_vectors_transform.py`. Data: `bench/results/s3_vectors_transform.csv`.
Consumed by [`docs/s3-vectors-recipe.md`](../../docs/s3-vectors-recipe.md).

**Setup**: 10,000 SPECTER2 embeddings (768-d), 100 held-out queries, 8 seeds
(99, 1–7). Seed 99 is `sketch_matryoshka.py`'s seed, so its rows are directly
comparable to `SKETCH_MATRYOSHKA.md`. Ground truth is the top-10 by
full-768-d float32 similarity; `R@N` is the fraction of that top-10 returned
in the top-N. `±` is the standard deviation across seeds.

## Why this exists

`SKETCH_MATRYOSHKA.md` measured center+truncate at k=256 and got
R@100 = 0.943. That measurement used an **inner-product** scan. S3 Vectors —
and every managed ANN service shaped like it — stores `float32` and offers
`cosine` or `euclidean`, never raw IP. So 0.943 describes a ranking function
the target index does not compute, and carrying it onto a cosine index would
be transferring a number across a metric change.

The centering question is the one at stake. Centering is the single biggest
lever on the binary path (+0.324 R@100 at k=64, `SKETCH_MATRYOSHKA.md`
finding 2), and `Corpus.build(center=True)` exists to make it automatic. The
natural assumption is that it carries over. It does not.

## Reproduction check

Two rows from `SKETCH_MATRYOSHKA.md` reproduce exactly at seed 99, which is
what licenses reading the rest of this table as a continuation of that one:

| row | SKETCH_MATRYOSHKA.md | here (seed 99) |
|---|---|---|
| `f32-centered`, k=256, IP, R@100 | 0.943 | 0.943 |
| `f32-raw`, k=256, IP, R@100 | 0.882 | 0.882 |

## Result

Against full-768 float32 **IP** ground truth, 8 seeds:

| transform | metric | dim | R@10 | R@100 |
|---|---|---:|---|---|
| truncate | ip | 256 | 0.413 ±0.011 | 0.887 ±0.013 |
| truncate | **cosine** | 256 | **0.703 ±0.014** | **0.998 ±0.001** |
| center+truncate | ip | 256 | 0.470 ±0.011 | 0.935 ±0.006 |
| center+truncate | cosine | 256 | 0.661 ±0.015 | 0.987 ±0.004 |

Two things move, and they move independently.

**1. Switching the metric from IP to cosine is worth more than any transform
choice.** At 256-d, truncate-only goes 0.887 → 0.998 R@100 purely by scoring
cosine instead of IP over the same stored bytes. SPECTER2 norms are tightly
clustered (mean 21.707, sd 0.146, cv 0.0067), so discarding them costs almost
nothing and removes a nuisance dimension from the comparison. The metric the
index computes was never a free variable here — it is fixed at cosine — so
this is the operating point, not an option.

**2. Centering is a net loss on this path.** Under cosine at 256-d,
truncate-only beats center+truncate by +0.042 R@10 and +0.011 R@100, on 8 of
8 seeds for both. Against a full-768 **cosine** ground truth the gap is wider
still (+0.094 R@10, +0.031 R@100, 8/8 at every dimension tested).

Centering head to head under cosine, positive = truncate-only wins:

| dim | ΔR@10 (IP gt) | ΔR@100 (IP gt) | ΔR@10 (cosine gt) | ΔR@100 (cosine gt) |
|---:|---|---|---|---|
| 128 | +0.018 (8/8) | +0.015 (8/8) | +0.057 (8/8) | +0.041 (8/8) |
| 192 | +0.026 (8/8) | +0.015 (8/8) | +0.072 (8/8) | +0.034 (8/8) |
| 256 | +0.042 (8/8) | +0.011 (8/8) | +0.094 (8/8) | +0.031 (8/8) |
| 384 | +0.050 (8/8) | +0.006 (8/8) | +0.119 (8/8) | +0.027 (8/8) |
| 512 | −0.015 (1/8) | +0.002 (7/8) | +0.164 (8/8) | +0.025 (8/8) |
| 768 | +0.004 (5/8) | +0.003 (7/8) | +0.268 (8/8) | +0.023 (8/8) |

The one row that goes the other way is dim=512 R@10 against IP ground truth
(−0.015, centering wins 7 of 8 seeds). It is recorded rather than smoothed
over, but it is not the recommended operating point and it does not survive
the switch to cosine ground truth, where 512-d is centering's *worst*
R@10 deficit of the whole sweep.

## Why centering inverts between the two paths

Centering helps the sign-bit path because sign bits are computed about the
origin: an uncentered coordinate whose mean is far from zero produces the
same bit for nearly every record and carries no information. Moving the
origin to the corpus mean is what makes those bits discriminative at all.
SPECTER2 has exactly this pathology — one dimension has mean ≈ 15.5.

A float32 cosine index has no such degenerate coordinate. It already
discards magnitude by normalizing, which is the same invariance centering was
being used to buy, and it reads the coordinates at full precision rather than
through a threshold. What centering does add is a shift of the origin away
from the one the ground truth is defined about, so the retained angle is
measured about the wrong point. `SKETCH_MATRYOSHKA.md` finding 3 describes
the same mechanism from the other direction — Hamming-on-centered tracks raw
IP better than centered-IP does.

So this is not a contradiction of the centering result. It is the boundary of
it: centering is a fix for thresholding at the origin, not a general-purpose
preprocessing step, and the float32 cosine path does not threshold.

## Stage 2 is exactly the stage-1 R@100

`rerank R@10` in the driver output equals stage-1 R@100 in every row, to
three decimals. This is not a coincidence worth measuring twice: stage 2
re-scores the returned candidates by the *same* full-768 float32 IP that
defines the ground truth, so it orders the true top-10 correctly whenever
they are present in the candidate set, and cannot recover one that is absent.
Stage-1 R@100 is therefore an exact ceiling on stage-2 R@10, and the ceiling
is met.

The practical consequence for the recipe: at 256-d cosine, R@100 = 0.998
means a stage-2 rerank returns R@10 = 0.998. The remaining 0.002 is candidates
stage 1 never retrieved, so a better reranker cannot help — only a wider
`topK` or more dimensions can.

## Caveats

Same as everything else in this directory: 10,000 SPECTER2 embeddings, 0.01%
of the full S2 corpus, one encoder. Nearest-neighbour distributions shift at
100M scale. The centering direction is consistent across 8 seeds and 6
dimensions, which makes it a solid claim about *this* corpus and encoder, not
a law. An encoder with widely varying norms would move the IP-vs-cosine
comparison; the cv here is 0.0067, and that number is the reason cosine is
nearly free.
