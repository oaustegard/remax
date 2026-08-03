# Fine-tuning LFM2.5 on a 4-core CPU box — feasibility, retrieval, and the frozen alternative

**Question:** every result in [LFM25_SUMMARY.md](LFM25_SUMMARY.md) uses the stock
encoder. Adapting the encoder itself is the other axis. Is it affordable here, and
does it pay?

**Answer: affordable but slow (10.8 h LoRA / 14.8 h full for the cookbook's 3×5k
recipe on CPU), and on a 221-pair retrieval task it overfits by epoch 2.** One
epoch is better than three on held-out queries. The frozen-backbone path costs
2.6 h for the same recipe and moves every subsequent iteration to sub-second.

Three separate runs, all on the same 4-core / 15 GB CPU box, all fp32:

| run | source | script |
|---|---|---|
| step-time feasibility | [`lfm25_finetune_feasibility.json`](lfm25_finetune_feasibility.json) | `bench/finetune_feasibility_lfm25.py` |
| retrieval fine-tune | [`lfm25_finetune_retrieval.json`](lfm25_finetune_retrieval.json), [`_1epoch.json`](lfm25_finetune_retrieval_1epoch.json) | `bench/finetune_retrieval_lfm25.py` |
| frozen-backbone classifier | [`lfm25_frozen_classifier.json`](lfm25_frozen_classifier.json) | `bench/frozen_classifier_lfm25.py` |

## 1. Is it even feasible? — measured step time, not a vibe

One real forward+backward+step at batch 8 × 512 tokens, 10 labels, extrapolated to
the classification cookbook's recipe (3 epochs over 5,000 documents).

| mode | trainable | % | s/step | ex/s | 3 epochs × 5k |
|---|--:|--:|--:|--:|--:|
| full | 354,494,218 | 100.0% | 28.44 | 0.28 | **14.81 h** |
| LoRA r=16 | 1,026,058 | 0.289% | 20.69 | 0.39 | **10.78 h** |
| frozen backbone | 10,250 | 0.0029% | 4.93 | 1.62 | **2.57 h** |

**LoRA is only 1.37x faster than a full fine-tune here**, which is the number most
likely to be assumed wrong. LoRA cuts the optimizer state and the gradient
memory, not the backward pass — activations still flow through the whole
backbone. And LFM2's architecture makes the ratio worse than usual: **10 of 16
layers are convolutional**, and PEFT can only wrap `Linear`, so the adapters reach
6 layers' worth of attention projections and nothing else. Freezing the backbone
outright — no backward through it at all — is what actually buys the 5.8x.

**Caveat on the RSS column.** All three rows report a peak RSS of 13.68 GB,
identically, because the harness reads `getrusage(RUSAGE_SELF)` — a *process-wide*
high-water mark — and the three modes run in one process with `full` first. So
13.68 GB is the full fine-tune's peak (against the box's ~15 GB ceiling), and the
LoRA and frozen peaks are **not measured** by this run. Do not cite them.

**Caveat on the hours column.** It is an extrapolation from 3 timed steps after
one warmup step, at a single fixed shape, with no data loading, no evaluation and
no checkpointing counted. Treat it as an order-of-magnitude answer to "can this
box do it overnight" — which is what it was asked for. CPU timing on this box
varies by ~25% run to run; a second full-fine-tune measurement recorded in
`bench/frozen_classifier_lfm25.py`'s docstring came in at 18.4 h against this
run's 14.8 h. The ~6x ratio between full and frozen was stable across both.

## 2. Retrieval fine-tune — the overfit is the result

SciFact has 339 judged (query, document) pairs across 300 queries. After a
seeded **query-level** split holding out 100 queries, training is **200 queries /
221 pairs**. That is roughly two orders of magnitude less data than the compute
budget suggests, so overfitting, not compute, is the thing to measure.

Setup: LoRA r=16 on attention *and* MLP projections (5,996,544 trainable),
in-batch-negatives cross-entropy, batch 8 (so 7 negatives per anchor — weak; real
recipes use 64+, and nothing bigger than 8 fits in 15 GB here), lr 1e-4, temp 0.05,
query len 64 / doc len 384, seed 42. Evaluation corpus is fixed for both models:
every judged document plus random distractors to 2,000 docs.

The split is at query level precisely because a document-level split leaks — the
same abstract can be relevant to a training query and a validation query.

> Absolute nDCG here is **not** comparable to the 0.7122 in
> [LFM25_SUMMARY.md](LFM25_SUMMARY.md): that is 300 queries against 5,183
> documents, this is 100 held-out queries against 2,000. Only the deltas within
> this table mean anything.

### fp32 nDCG@10 — train vs held-out

| | train (200 q) | val (100 q, held out) | train − val |
|---|--:|--:|--:|
| base model | 0.7931 | 0.8017 | −0.0086 |
| **fine-tuned, 1 epoch** | 0.8902 | **0.8387** (+0.0370) | +0.0516 |
| fine-tuned, 3 epochs | 0.9132 | 0.8332 (+0.0314) | **+0.0800** |

Training loss over the three epochs: 0.0714 → 0.0439 → **0.0086**. Training
nDCG climbs monotonically to 0.9132. Held-out nDCG **peaks at one epoch and then
falls**, and the train-minus-val gap goes from −0.0086 (base, i.e. the held-out
queries were slightly *easier*) to +0.0800. Epochs 2 and 3 cost 15 minutes and
bought nothing on the only queries that count.

An eightfold drop in training loss with a *declining* validation metric is the
textbook shape, and on 221 pairs it should be the expected one. If you run this,
run one epoch and stop.

### The quantized index behaves differently from fp32

| val (held out) | base | 1 epoch | 3 epochs |
|---|--:|--:|--:|
| fp32 | 0.8017 | **0.8387** | 0.8332 |
| 1-bit symmetric (128 B/vec) | 0.7789 | 0.8183 | **0.8277** |
| 1-bit asymmetric (128 B/vec) | 0.7806 | 0.8188 | **0.8286** |
| quantization penalty (fp32 − sym) | 0.0228 | 0.0204 | **0.0055** |
| agree@10, symmetric | 0.701 | 0.751 | **0.780** |

The epoch that hurts fp32 held-out retrieval keeps helping the 1-bit index. By
epoch 3 the quantization penalty has collapsed from 0.0228 to 0.0055 and top-10
agreement with fp32 has risen from 0.701 to 0.780 — the fine-tune made the
representation *easier to binarize* while making it slightly worse in float.

That is a plausible mechanism — a contrastive objective at temperature 0.05
pushes pairs apart in angle, which is exactly the quantity sign bits preserve —
but this run does not establish it. On 100 queries the 1-bit gain from epoch 1 to
3 (+0.0094) is about twice the resolvable band and the fp32 loss (−0.0055) is
about one band. Read it as a hypothesis with one supporting run, not a finding.
It would be worth a real test: it implies quantization-aware fine-tuning may need
no quantization-aware *loss* at all.

### Cost

23.5 minutes for 3 epochs (8.5 / 7.5 / 7.5), plus a re-encode of the 2,000-doc
evaluation corpus with the fine-tuned model. That is ~18 s per step over 28 steps
per epoch, against the 20.69 s/step the feasibility run measured for LoRA at
batch 8 × 512 tokens — two independent measurements of the same architecture
agreeing to within the shorter sequence lengths used here.

## 3. The frozen alternative — the cookbook, cheaply

Same pipeline as Liquid's classification cookbook (mean-pool, linear head,
BCE-with-logits, per-label threshold tuning) with the backbone frozen and its
output cached:

    encode once  →  train a head on cached vectors  →  tune thresholds  →  eval

Head training: **0.191 s** for 30 epochs. The cookbook's full fine-tune of the
same shape measured 14.8 h (§1). Everything after the encode — head
architecture, class weighting, threshold sweeps — becomes free, where the
fine-tuning recipe pays the whole forward+backward again each time.

**These numbers are from the script's `--demo` path: 600 train / 200 test
*synthetic* tickets over 4 label-correlated classes.**

| test set | micro-F1 | macro-F1 |
|---|--:|--:|
| fixed threshold 0.5 | 0.9239 | 0.9240 |
| per-label tuned thresholds | 0.9934 | 0.9931 |

The synthetic task is easy by construction — templated fragments drawn per label —
so **the absolute F1 is a smoke test, not a classification benchmark**, and nothing
should be concluded from it about LFM2.5 as a classifier. What the run does show
is that per-label threshold tuning is worth ~7 points of micro-F1 over a fixed 0.5
(thresholds landed at 0.37 / 0.41 / 0.28 / 0.40, none near 0.5), and that the
tuning stage costs nothing on cached vectors. Thresholds are fitted on *training*
predictions and applied to test, so the test number is not self-tuned.

The one-time encode is the whole cost, and it is real: `bench/frozen_classifier_lfm25.py`'s
docstring records ~83 min for 5,183 documents at 512 tokens on this box. That
figure is not in any committed JSON — it is a run note, cited here as such.

## What this means for remax

Nothing in the library changes. remax compresses whatever vectors it is given,
and all three runs are about the encoder producing them. The findings are
recorded because they bound what an "adapt the encoder" answer costs, and because
§2's quantization-penalty collapse is the one result here that touches remax's
own claim.

Practical ordering, on CPU:

1. **Freeze and cache** unless the text is genuinely out of distribution. 2.6 h,
   then free iteration.
2. **If you must fine-tune retrieval on a few hundred pairs, run one epoch.**
   Held-out quality peaks there.
3. **Do not reach for LoRA expecting speed.** It buys 1.37x here; the backward
   pass through 354M frozen-but-differentiated parameters is what costs.

## Scope and limits

- One box, one encoder, fp32, CPU only. Every wall-clock figure is
  box-specific and varies ~25% run to run.
- §1's hours are extrapolated from 3 timed steps; §1's RSS is a single
  process-wide peak misattributed to three rows in the raw JSON (see caveat).
- §2 is 100 held-out queries against a 2,000-doc corpus, one seed, one
  learning rate, batch 8. No sweep. Differences under ~0.005 nDCG are not
  resolvable, and the interesting deltas are only 2-4x that.
- §2 tested only LoRA r=16. A full fine-tune on 221 pairs was not run; on this
  data it would be expected to overfit harder, not less, but that is an
  expectation, not a measurement.
- §3's F1 numbers are synthetic-demo only. The encode timing is a docstring run
  note, not committed data.
