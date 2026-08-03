# Can a learned rotation beat a random one at the same bits? — BEIR SciFact, LFM2.5

**Question:** remax rotates with a Haar-random orthogonal matrix before taking
signs. That rotation is data-oblivious — reproducible from `(d, seed)`, needs no
corpus — but by construction it is not tuned to the data it is about to throw
information away about. What does a data-*dependent* rotation buy at identical
storage?

**Answer: about +0.020 symmetric nDCG@10 for free bits and a 4.19 MB side-car,
which is worth less than the asymmetric scoring you can have for nothing — and
one of the four transforms tested fails in a way worth keeping.** The learned map
with the *lowest* quantization error had the *worst* retrieval, because it stopped
being a rotation.

Corpus: BEIR SciFact, 5183 docs / 300 test queries / 339 judgments, LFM2.5-Embedding-350M
at fp32, d=1024 → **128 B/vec for every row below**. Single seed (42). Source:
[`lfm25_learned_rotation.json`](lfm25_learned_rotation.json) and
[`lfm25_learned_rotation_holdout.json`](lfm25_learned_rotation_holdout.json),
written by `bench/learned_rotation_lfm25.py`. fp32 reference at 4096 B/vec:
nDCG@10 **0.7122**.

Every transform is fitted on **document embeddings only**. Queries and qrels are
never seen during fitting, so the qrels evaluation is held out with respect to the
thing being learned.

## The four transforms

| | what it optimizes | constrained orthogonal? |
|---|---|---|
| **haar** | nothing — Haar-random, the current remax default | yes, by construction |
| **itq** | `‖B − XR‖_F` with `B = sign(XR)`, alternating sign() and orthogonal Procrustes (Gong & Lazebnik, CVPR 2011) | yes, SVD each iteration |
| **ste** | fp32 ranking distilled into the binary code, triplet margin, straight-through through `sign()`, free `W` | **no** |
| **ste-orth** | same loss as `ste`, but `W = R₀·Cayley(P − Pᵀ)` so every reachable point is a rotation | yes, structurally |

## Result — fit on all 5183 documents

| method | fit s | quant err | orth err | symmetric | vs haar | asymmetric | vs haar |
|---|--:|--:|--:|--:|--:|--:|--:|
| haar | 0.1 | 976.98 | 1.1e-05 | 0.6772 | — | 0.7020 | — |
| **itq** | 13.3 | 972.60 | 1.9e-05 | **0.6972** | **+0.0200** | **0.7068** | **+0.0048** |
| ste | 13.4 | **922.23** | **98.41** | 0.4967 | −0.1805 | 0.4996 | −0.2025 |
| ste-orth | 30.6 | 976.97 | 2.6e-05 | 0.6825 | +0.0052 | 0.6976 | −0.0044 |

ITQ is the only transform that wins on both scoring paths, and its symmetric
gain is the largest effect in the table by a factor of four. It costs 13 s of
fit on this corpus, is closed-form, has no learning rate and no early stopping,
so there is no hyperparameter to overfit.

`ste-orth`, which is the *same objective as `ste`* restricted to the rotation
manifold, recovers almost all of the damage and lands within noise of Haar
(+0.0052 symmetric, −0.0044 asymmetric, on a 300-query set where differences
under ~0.005 are not resolvable). Constrained gradient descent on the ranking
loss buys nothing over the closed-form Procrustes solution and costs 2.3x the
fit time.

## The negative result: quantization error is a proxy, and a free map wins the proxy

`ste` reached a **lower quantization error than ITQ** (922.23 vs 972.60 — the
best of anything tested, including the transform whose entire objective *is*
quantization error) and the worst retrieval by a margin of 0.18 nDCG. Its
orthogonality error `‖WᵀW − I‖_F` is **98.4**, against ~2e-05 for everything
else. The learned map stopped being a rotation.

The mechanism is not subtle once the number is in front of you: nothing stops an
unconstrained linear map from collapsing the space. Shrink the coordinates that
are expensive to sign correctly, and both `‖sign(XW) − XW‖` and the margin loss
improve while the retrieval geometry is destroyed. Agreement with the fp32 top-10
falls to 0.376, from 0.701 for Haar — the codes are no longer ranking the same
corpus.

Two things follow, and they are the durable content of this file:

1. **Do not tune a rotation against reconstruction error alone.** On this corpus
   the transform with the lowest quantization error had the worst nDCG, and the
   ordering of the two metrics is *inverted* across the four rows. Report
   orthogonality error next to any learned transform; it is one line
   (`‖WᵀW − I‖_F`) and it is the thing that catches this.
2. **Projecting back to the orthogonal group periodically does not rescue it.**
   That was tried first: `W` becomes ill-conditioned fast enough that the SVD
   fails to converge before the first projection lands. The structural
   constraint (Cayley) is what works, and it works by never leaving the manifold —
   `P` initializes at zero, so training starts exactly at the Haar baseline.

## Holdout — part of ITQ's gain is memorization

The same run with `--holdout`: fit on a random half of the documents (2591 of
5183), evaluate on all of them.

| method | fit on | symmetric | vs haar | asymmetric | vs haar |
|---|---|--:|--:|--:|--:|
| haar | — | 0.6772 | — | 0.7020 | — |
| itq (full fit) | 5183 docs | 0.6972 | +0.0200 | 0.7068 | +0.0048 |
| itq (holdout fit) | 2591 docs | 0.6914 | +0.0142 | 0.6978 | **−0.0042** |

The symmetric gain survives at about 70% of its size — most of it is corpus
geometry the transform genuinely learned. The **asymmetric gain does not survive
at all**; it goes negative. So the part of ITQ's advantage that showed up on the
float-query path was fitting the specific vectors it was then scored on, and a
deployment that indexes documents it did not fit on should expect the asymmetric
column to read as no better than Haar.

The transfer risk registered before the run was different — that SciFact queries
are short claims and documents are long abstracts, so a document-fitted rotation
might not fit the query distribution, showing up as itq winning symmetric and
losing asymmetric. Under the full fit that did not happen. Under holdout it did.

## Why this is not shipped

Three reasons, in the order that decides it.

**1. The free lever is bigger.** Asymmetric scoring — keep the query in float,
binarize only the index — buys **+0.0248** symmetric→asymmetric on this same
codec at this same 128 B/vec ([LFM25_SUMMARY.md](LFM25_SUMMARY.md#asymmetric-scoring)),
costs no storage and no fit. ITQ's symmetric gain is +0.0200, *smaller*, and the
two are almost entirely non-additive: stacking ITQ on top of asymmetric scoring
adds only +0.0048 (and −0.0042 on holdout). Both mechanisms are buying down the
same thing — the variance of a similarity estimated from sign bits — and
asymmetry has already taken that slack, exactly as stacking does.

**2. It costs more than the index it improves, below ~33k vectors.** A learned
rotation is not reproducible from a seed; the matrix must ship with the index.
At d=1024 that is 1024×1024 fp32 = **4,194,304 B = 4.19 MB**, which is
**32,768 vectors' worth of codes** at 128 B/vec. SciFact's 5183-document index
is 663 KB, so the rotation would be **6.3x larger than the index it improves**.
The crossover is a fixed vector count, not a fixed fraction: below ~33k vectors a
learned rotation is a net storage loss no matter how much nDCG it buys. (Storing
it at fp16 would halve that to ~16k vectors — untested here, and it would perturb
the codes unless the same precision is used at build and query time.)

**3. It breaks the reproducibility contract, at the file-format level.** remax's
index is defined by `(d, seed)`: `Corpus`'s on-disk header stores exactly `n`,
`d`, and an int64 `seed`, with the rotation *construction* ("haar"/"rht") named in
a sidecar. There is nowhere to put a matrix. Shipping a learned rotation is a
disk-format change, not a parameter change — and it ends data-obliviousness,
which is a property remax currently has and states.

## Disposition

Not implemented in the library. `bench/learned_rotation_lfm25.py` is kept
runnable; this file is the record.

If someone revisits this, the thing that would change the answer is scale: at
≥ 100k vectors the 4.19 MB side-car is under 5% of the index, and reason (2)
stops binding. Reasons (1) and (3) do not scale away. Start from ITQ — closed
form, 13 s, monotone — and do not restart the straight-through path without
reading the `ste` row above.

## Scope and limits

- One dataset, one seed, one encoder, 300 queries. Differences under ~0.005 nDCG
  are not resolvable; `ste-orth` vs `haar` is inside that band on both paths.
- ITQ ran 30 iterations, `ste` 400 steps, `ste-orth` 300 steps. No sweep over any
  of those; a longer `ste-orth` run is not ruled out, only unmeasured.
- Fit times are 4-thread CPU on the bench box and are not a benchmark of anything
  — they are here to show ITQ is cheap enough that cost is not the objection.
- The holdout run only tested `haar` and `itq`. `ste` and `ste-orth` were not
  re-run under holdout, because the full-fit result already disqualified them.
