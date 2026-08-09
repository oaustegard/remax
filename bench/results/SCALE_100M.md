# One query at n = 1e8

**This is a benchmark, not a gate**, on the same terms as
[`QUERY_PATH_SPEED.md`](QUERY_PATH_SPEED.md): there is no published constant
for how fast a Hamming scan should be, so none of these numbers can be gated.
What *is* gated is the only claim with an anchor — that none of this changes a
single neighbour — by `bench/gates/query_path_gate.py`, which proves itself by
going red under 16 simulated defects.

`QUERY_PATH_SPEED.md` closes by saying it establishes **nothing about a corpus
larger than n = 1e7**, and means it. This file measures the range it declined
to extrapolate into, because that range is where the question in
[issue #70](https://github.com/oaustegard/remax/issues/70) lives.

Reproduce with `python3 bench/scale100m.py --trials 5`. Raw output in
[`scale100m.json`](scale100m.json).

## The question

The blog post *Three Gigs to Search a Hundred Million Papers* claims that
single-threaded brute force returns top-100 candidates in "well under a
second". At 100M x 32 B = 3.2 GB that is now measurable rather than
extrapolated. Issue #70 measured a **standalone C kernel** and found the scan
comfortably inside the budget and selection blowing it — `np.argpartition`
costing roughly 2x the scan — and asked three things of remax's own path,
which is not that kernel:

1. what does `hamming_distances` + `stable_top_k` cost end to end at n≈1e8;
2. does the existing blocked path already close the gap for one query;
3. is a threshold-fused variant worth carrying.

Short answers: **0.81x the scan, not 2x** — the difference is that remax
dispatches to a counting select above `COUNTING_MIN_N` and never runs the
argpartition the issue measured. **No**, blocking alone does essentially
nothing for one query, and it was not even reachable automatically. **Yes**,
but not in the two-phase sample-a-cutoff shape the issue proposed; the useful
form needs nothing sampled and nothing estimated.

## The box

| | |
|---|---|
| CPU | Intel(R) Xeon(R) Processor @ 2.80 GHz, 4 cores / 4 threads, 1 socket |
| Cache | L1d 128 KiB (4x), L2 4 MiB (4x), L3 33 MiB (1 instance) |
| Memory | 16,461,004 kB (~15.7 GiB) |
| OS | Linux 6.18.5, x86_64, KVM guest |
| Python | 3.11.15 |
| NumPy | 2.4.4 |
| remax | 0.2.0 + this branch |
| Native kernel | available (gcc, `-O3 -mpopcnt`) |

**This box scans at ~6.2 GB/s single-threaded at n=1e8; issue #70's standalone
kernel measured 10.7 GB/s.** So every absolute figure here is on a slower
machine than the one that opened the issue, and the two sets of numbers should
not be put in the same table. Ratios within this file are comparable to each
other; nothing here rescales onto that box, and no attempt is made to.

**Method.** Minimum of 5 timed runs after a warmup. Minimum rather than mean:
the distribution is a hard floor (the work) plus a one-sided noise tail
(scheduling, page faults, co-tenants on a KVM guest).

**Warm, and it matters more here than anywhere else in this repository.**
Issue #70 first measured argpartition at **5391 ms** and warm at 714 ms — 7.6x
apart, the difference being page faults on a freshly written 400 MB score array
plus an 800 MB permutation allocation. A cold number would have made selection
look catastrophic rather than merely dominant, and the 5391 ms figure nearly
shipped. Everything below is warm.

**Between-run variance is larger than min-of-5 suggests.** Three whole-file
runs while preparing this measured the n=1e8 single-threaded scan at 7.14,
6.99 and 6.18 GB/s — a 16% spread on the same code and the same box. The
tables below are all from the run that produced the committed
`scale100m.json`, so markdown and JSON cannot drift apart, but read the
speedups as "roughly 1.5-1.8x", not as three significant figures.

## Matched implementation effort

A gate cannot check comparability, so each comparison states what its arms
share. Every arm below scans the identical corpus with the identical C kernel
and performs the identical `n * B` popcount operations; they differ only in
what happens to the distances afterwards.

| Comparison | Why the arms are comparable |
|---|---|
| scan vs anything | The scan arm is the *floor*: `hamming_distances` into a reused buffer, no selection at all. Nothing below it can be beaten by it. |
| unblocked vs blocked | Same kernel, same block-visiting order, same merge. The unblocked arm is the shipped pre-#70 code path, reached through the same function. |
| per-block top-k vs running threshold | Identical blocking, identical merge, identical scan. One step differs: rank every row of the block, or test it against the k-th distance already held. |
| counting select vs argpartition | Both NumPy, both exact, both asserted to return the byte-identical permutation and to equal `np.argsort(kind="stable")[:k]`. The argpartition arm is `stable_top_k`'s own comparison branch lifted out verbatim, widening included — not a strawman. |

Every row in every table was verified against `np.argsort(kind="stable")[:k]`
on every trial. A speed comparison between a right answer and a wrong one
measures nothing.

The corpus is uniform random bytes, so distances are `binomial(8B, 0.5)` —
257 possible values at B=32, tie-dense, which is the shape a real Hamming scan
produces. k=100 throughout.

---

## 1. End to end, one query

B=32, k=100, one thread. `unblocked` is the pre-#70 path; `blocked` is what
ships now.

| n | scan (floor) | unblocked, fresh out | unblocked, reused `out=` | unblocked + argpartition | **blocked** | speedup |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000,000 | 4.94 ms | 8.00 ms | 8.23 ms | 8.12 ms | **6.32 ms** | 1.27x |
| 10,000,000 | 53.9 ms | 97.9 ms | 85.6 ms | 131.2 ms | **57.9 ms** | 1.69x |
| 100,000,000 | 517.5 ms | 937.9 ms | 795.8 ms | 1,437.6 ms | **566.7 ms** | 1.66x |

Selection as a multiple of the scan it follows — the number the issue was
actually about:

| n | unblocked | blocked |
|--:|--:|--:|
| 1,000,000 | 0.62x | 0.28x |
| 10,000,000 | 0.82x | **0.075x** |
| 100,000,000 | 0.81x | **0.095x** |

And on four threads:

| n | scan | unblocked | blocked | speedup |
|--:|--:|--:|--:|--:|
| 1,000,000 | 1.62 ms | 4.67 ms | 4.31 ms | 1.08x |
| 10,000,000 | 16.1 ms | 46.6 ms | 26.1 ms | 1.79x |
| 100,000,000 | 145.5 ms | 441.3 ms | **250.6 ms** | 1.76x |

**Answer to question 1.** At n=1e8, one query, one thread: **937.9 ms** with a
fresh `(n,) int32` output array, **795.8 ms** reusing one. The scan is 517.5 ms
of that, so selection costs **0.81x the scan** — not the 2x the issue
measured, because remax is above `COUNTING_MIN_N` and runs the counting select,
not `argpartition`. Run the argpartition arm and the issue's number reproduces
exactly: 1,437.6 − 517.5 = 920 ms, or **1.78x the scan**.

So the disagreement was never about the scan or about the machine. It was
about which selection the path in question runs, and remax already had the
better one.

**On the blog post's claim.** Single-threaded top-100 at n=1e8 on this box was
937.9 ms before this change and is 566.7 ms after. "Well under a second" was
already true on the shipped path, with about 60 ms of margin; it now has 430 ms
of margin. It is *not* true of the argpartition path, at 1,437.6 ms. Which
sentence is correct depends entirely on which implementation a deployment runs
— which is what made question 1 worth asking rather than assuming.

**The 400 MB that stopped being allocated.** The unblocked path materialises
one `(n,) int32` score array — 400 MB at n=1e8, every element written, ~100 of
them ever read. The reused-`out=` column is what a caller gets for hoisting
that allocation out of a query loop: 142 ms at n=1e8, 15% of the total. The
blocked path allocates 4 MiB and needs no hoisting.

---

## 2. Per-block top-k vs the running threshold

Both arms block; both use the same merge. The old one calls `stable_top_k` on
every block, to rank all its rows and find the k that might matter. The new one
keeps the k-th distance of the candidates already held and tests each block
against it in a single comparison pass.

| n | block rows | scores | codes | per-block top-k | running threshold | speedup |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000,000 | 32,768 | 0.125 MiB | 1 MiB | 9.28 ms | 6.37 ms | 1.46x |
| 1,000,000 | 131,072 | 0.5 MiB | 4 MiB | 9.85 ms | 6.40 ms | 1.54x |
| 1,000,000 | 250,000 † | 0.95 MiB | 7.6 MiB | 8.65 ms | 6.41 ms | 1.35x |
| 1,000,000 | 524,288 | 2 MiB | 16 MiB | 8.02 ms | 8.56 ms | 0.94x |
| 10,000,000 | 32,768 | 0.125 MiB | 1 MiB | 90.6 ms | 63.0 ms | 1.44x |
| 10,000,000 | 131,072 | 0.5 MiB | 4 MiB | 102.7 ms | 66.5 ms | 1.55x |
| 10,000,000 | 524,288 | 2 MiB | 16 MiB | 89.3 ms | 62.7 ms | 1.42x |
| 10,000,000 | 1,048,576 † | 4 MiB | 32 MiB | 94.2 ms | 65.0 ms | 1.45x |
| 10,000,000 | 2,097,152 | 8 MiB | 64 MiB | 95.3 ms | 70.0 ms | 1.36x |
| 10,000,000 | 8,388,608 | 32 MiB | 256 MiB | 88.7 ms | 81.0 ms | 1.10x |
| 100,000,000 | 32,768 | 0.125 MiB | 1 MiB | 1,004.0 ms | 665.9 ms | 1.51x |
| 100,000,000 | 131,072 | 0.5 MiB | 4 MiB | 1,077.1 ms | 607.7 ms | 1.77x |
| 100,000,000 | 524,288 | 2 MiB | 16 MiB | 830.2 ms | 581.2 ms | 1.43x |
| 100,000,000 | 1,048,576 † | 4 MiB | 32 MiB | 849.3 ms | **569.3 ms** | 1.49x |
| 100,000,000 | 2,097,152 | 8 MiB | 64 MiB | 889.0 ms | 606.3 ms | 1.47x |
| 100,000,000 | 8,388,608 | 32 MiB | 256 MiB | 837.8 ms | 601.6 ms | 1.39x |

† the block `_resolve_block` picks automatically for that n.

**Answer to question 2: no, twice over.**

First, blocking a single query was **not reachable** without asking for it.
`_resolve_block` returned `n` for `m < 2` on the argument — correct at the
time — that one query reads the corpus once whatever the block size. So the
blocked path could only be entered by passing `block=` explicitly, and nothing
in `Corpus.search` ever did.

Second, forced on, it did not help. At n=1e8 the per-block-top-k column runs
830-1,077 ms against 937.9 ms unblocked: it *straddles* the number it was
supposed to beat, and the automatic block would have given 849.3 ms, a 1.10x
that is inside this file's between-run variance. Blocking moves the corpus
traffic that was already minimal — with one query the corpus streams once at
any block size — and leaves the selection work exactly where it was. It is the
filtering, not the blocking, that pays.

**What the block is sized on.** For a batch, the block holds *codes* that get
reused by every query, so `_BLOCK_TARGET_BYTES` sizes it in code bytes. For one
query that reasoning does not apply, and the quantity to keep in cache is the
`(blk,) int32` the scan writes and the filter immediately reads back. Hence a
separate `_SINGLE_QUERY_SCORE_BYTES = 4 MiB`. The n=1e8 column is the evidence:
665.9 ms at 0.125 MiB of scores, 569.3 ms at 4 MiB, 601.6 ms at 32 MiB — a
plateau across roughly 0.5-8 MiB with both edges clearly worse, centred where
the score buffer fits this box's 4 MiB L2. 2 MiB and 4 MiB are not
distinguishable across runs (2 MiB measured better at n=1e7 here and in two
earlier runs; 4 MiB measured better at n=1e8 in two of three), so **the L2
argument, not a 3% difference, is what picks 4 MiB.** That is a design choice
resting on a measured plateau, not a measured optimum, and it is stated as one.

**The 0.94x row is real and worth keeping.** At n=1e6 with 524,288-row blocks
there are two blocks: the first establishes the threshold with a full
`stable_top_k` and the second is the only one that filters. There is not enough
corpus left to pay back the bookkeeping. That is exactly the case
`_MIN_SINGLE_QUERY_BLOCKS = 4` exists to avoid — the automatic block at n=1e6
is 250,000 rows, four blocks, and measures 1.35x.

---

## 3. Where the unblocked path's time went

Selection alone, on an already-materialised score array.

| n | counting select | of which histogram | argpartition | counting wins by | argpartition allocates |
|--:|--:|--:|--:|--:|--:|
| 1,000,000 | 3.99 ms | 2.05 ms (51%) | 3.01 ms | 0.75x | 8 MB |
| 10,000,000 | 36.8 ms | 22.7 ms (62%) | 91.5 ms | 2.49x | 80 MB |
| 100,000,000 | 307.3 ms | 216.1 ms (**70%**) | 801.6 ms | 2.61x | 800 MB |

The counting select's 2.5-2.6x over argpartition at 1e7 reproduces
`QUERY_PATH_SPEED.md`'s 2.43-2.63x, on a different day, which is a useful
consistency check on both files. It holds at 1e8.

The new information is the middle column. **At n=1e8 the histogram pass alone
is 70% of selection**, running at ~1.9 GB/s over an array the scan wrote at
6.2 GB/s. That is the cost the running threshold removes rather than optimises:
it never builds a histogram, because it never needs to know the exact cutoff —
only whether a given row is beyond the one already established.

This also explains why issue #70's counting-sort experiment did not help. Its
diagnosis — "at this scale the cost is the number of passes over n, not the
comparison strategy" — is right, and the conclusion follows: the only way to
win is to stop making passes over an `(n,)` array, which means never
materialising one.

---

## 4. `_COUNT_CHUNK` re-swept — measured, not changed

`_COUNT_CHUNK` is 2^20, chosen at n=1e7 from a sweep of 2^18-2^20
(`QUERY_PATH_SPEED.md` §2). Since the histogram is 70% of selection at n=1e8,
the constant is worth re-asking there.

| n | 2^14 | 2^16 | 2^18 | 2^20 (shipped) | 2^22 | best vs shipped |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000,000 | 2.71 ms | **2.33 ms** | 2.58 ms | 4.68 ms | 3.96 ms | 2.01x |
| 10,000,000 | 28.7 ms | **26.9 ms** | 29.9 ms | 38.0 ms | 73.0 ms | 1.42x |
| 100,000,000 | 274.0 ms | **227.9 ms** | 254.4 ms | 266.9 ms | 402.1 ms | 1.17x |

2^16 wins at every size measured here, and the shipped 2^20 is the worst of the
non-degenerate options at two of three. **Nothing in this PR changes it**, for
two reasons. It is a constant owned by a different benchmark — moving it means
re-running `QUERY_PATH_SPEED.md` §2, whose table would otherwise silently go
stale — and it is no longer on the single-query critical path above
`_MIN_SINGLE_QUERY_N`, since that path does not build a histogram at all. It
still matters below that floor, for `search_asymmetric`'s callers, and for
`StackedSignBitQuantizer.search`, which has its own m-loop (see *Scope*).

Recorded here so the next person does not have to re-measure it.

---

## 5. The m > 1 batch path, which the same change also touched

The running threshold replaced the per-block `stable_top_k` for **every** m,
not only for one query, and a batch is the shape that path was written for. So
it is measured rather than asserted to be fine. k=10 here, matching
`QUERY_PATH_SPEED.md` §3.

| n | m | threads | per-block top-k | running threshold | speedup |
|--:|--:|--:|--:|--:|--:|
| 1,000,000 | 2 | 1 | 17.3 ms | 11.3 ms | 1.54x |
| 1,000,000 | 2 | 4 | 15.7 ms | 9.39 ms | 1.68x |
| 1,000,000 | 8 | 1 | 55.5 ms | 38.7 ms | 1.43x |
| 1,000,000 | 8 | 4 | 38.2 ms | 19.2 ms | 1.99x |
| 1,000,000 | 32 | 1 | 211.7 ms | 155.8 ms | 1.36x |
| 1,000,000 | 32 | 4 | 140.2 ms | 75.4 ms | 1.86x |
| 10,000,000 | 2 | 1 | 179.4 ms | 110.3 ms | 1.63x |
| 10,000,000 | 2 | 4 | 156.4 ms | 81.5 ms | 1.92x |
| 10,000,000 | 8 | 1 | 558.0 ms | 370.0 ms | 1.51x |
| 10,000,000 | 8 | 4 | 387.9 ms | 204.5 ms | 1.90x |
| 10,000,000 | 32 | 1 | 2,127.0 ms | 1,433.8 ms | 1.48x |
| 10,000,000 | 32 | 4 | 1,337.6 ms | **610.4 ms** | 2.19x |

Every row identical. No regression, and a 1.4-2.2x that was not the point of
the exercise — the per-block rank was pure overhead for batches too, it was
simply less visible there because the corpus traffic dominated.

---

## Answering question 3, and what was *not* built

**Yes, a fused threshold is worth carrying — the version with no threshold
estimator in it.**

Issue #70 proposed the two-phase shape: sample or histogram a prefix, pick a
cutoff, run one fused pass, fall back when the cutoff under-fills. That was
prototyped during this work and it does work — at n=1e8 it landed within the
between-run noise of what shipped, which is to say the two are not
distinguishable on this box. (Those prototype numbers are **not** in
`scale100m.json`; the driver was deleted rather than carried, and the claim
here is "no better", not a measurement anyone should quote.) It was not built,
because everything it needs is machinery the running threshold does not:

* a cutoff estimator, whose accuracy is a statistical claim about the tail of
  a distribution — top-100 of 1e8 is the 1e-6 quantile, and a 1% prefix sees
  about one row that qualifies;
* a sampling strategy that is not a prefix, because a corpus clustered by
  topic makes its first 1% a biased sample of distances to any given query;
* an under-fill fallback, which is a second full scan, plus a test for it;
* a cap on the candidate set, because a corpus of near-duplicates puts an
  unbounded number of rows under any cutoff.

The running threshold needs none of that. The bound *is* the k-th distance of
the candidates already held; it starts at 8B, where it filters nothing, and
falls monotonically as better candidates arrive. It cannot under-fill, because
it never discards a row that could have entered — a row further away than the
k-th already has k rows at least as close and earlier in index order. There is
no estimate to be wrong and no fallback to test.

That last property is why this is not the candidate pruning remax's anti-goals
rule out. Nothing is pruned from the *scan*: every row of the corpus is still
scored by the same kernel, and `n * B` popcount operations still happen. What
is skipped is only the ranking of rows that provably cannot be in the answer,
which is why the output is bit-identical rather than approximately right.

### The comparison sense, and a defect that turned out not to be one

The filter is `d < thresh`, strict — the opposite of what the tie-stability
rule everywhere else in the query path would suggest. A simulated `<=`→`<`
defect was written for the gate on the assumption that dropping the boundary
tie group was the remax#32 failure again. It came back green, and it was right
to: the caller reaches the filter only when it holds exactly k candidates, so a
row tying the k-th has a larger index and loses `(distance, index)` order
regardless. Admitting the tie group is equally correct and pure cost — and on a
corpus of identical rows, `<=` selects every row of every block, which is the
`O(n)` materialisation the path exists to avoid.

The simulation was deleted rather than kept as a passing check, and the gate
states in its coverage limits that the comparison sense is not gated because
neither sense is wrong. Two defects that *are* real replaced it: a threshold
read from the front of the candidate list rather than its k-th slot, and a
per-block trim cut with `argpartition` instead of a stable sort.

Writing the second of those exposed something the self-test would otherwise
have hidden. `blocked-merge-forgets-earlier-blocks`, a known-bad that had been
red since v0.2.0, went **green** under the new code — the filter made its guard
(`this block already has k candidates, why merge?`) almost never fire, so the
defect became inert rather than caught. An optimisation quietly retiring an
existing gate case is not a hypothetical; this one did.

Restoring it needed a corpus built for the purpose, and the first attempt did
not work. A heavily duplicated corpus is tie-dense but its threshold collapses
immediately, so blocks never overflow the k-list and the trim never runs — the
check passed and caught nothing. What works is 2,000 far rows followed by five
distinct near rows repeated 800 times: the far prefix leaves the threshold
loose so later blocks put hundreds of rows under it, and the near tail puts the
trim's k-th position inside a tie group. The gate counts how many times the
trim ran and how many of those had a boundary tie, and fails if that count is
zero, because a check that never reaches the code it names is worse than no
check.

That whole detour was found by the self-test, but only after a second bug was
fixed: a local variable in the new check shadowed one the gate used later, and
the resulting crash read as `exit=1` — which the self-test scores as a defect
correctly rejected. Every simulated defect reported red while two of them were
in fact green. **A red from a crash is not a red from a catch**, and the only
thing that separated them was running the gate without `--simulate`.

---

## Scope, and what none of this establishes

* **One box, one code width, one k.** B=32 (d=256) and k=100 throughout
  sections 1-4, k=10 in section 5. A wider code moves the alphabet (2049
  values at B=256) and a larger k makes the per-block trim and the merge more
  expensive; neither was swept.
* **One distribution.** Uniform random bytes, so distances are
  `binomial(256, 0.5)` — tightly concentrated, tie-dense. A corpus with a very
  different tie structure, or one where the query is close to a large cluster,
  changes how many rows survive each block, which is the quantity the whole
  gain rests on. The degenerate ends are tested for *correctness*
  (`tests/test_query_speed_paths.py`) but not for speed.
* **Block size was swept; `_MIN_SINGLE_QUERY_N` and
  `_MIN_SINGLE_QUERY_BLOCKS` were not.** The floor at 2^16 comes from a
  separate sweep at n=1e4-1e6 that found the win appearing at n=1e5 (1.33-1.45x)
  and absent below; that sweep is not in `scale100m.json`. The `>= 4 blocks`
  rule is justified by the 0.94x row in section 2 and by nothing else.
* **`StackedSignBitQuantizer.search` gets none of this.** It has its own
  `hamming_distances` + `stable_top_k` loop and does not route through
  `hamming_topk_batch`, so the k-stack path still pays full selection at every
  n. That divergence predates this issue and is not fixed here.
* **`search_asymmetric` gets none of this either**, and cannot as written: its
  scores are floats, where the "bounded integer alphabet" half of the argument
  does not apply. The threshold itself would still work — it needs only a
  total order — but that is an untested claim, not a measurement.
* **Nothing about other hardware.** Four cores, one socket, ~6-7 GB/s
  single-threaded and ~23 GB/s aggregate. The threading speedups are bounded
  by the memory system rather than the core count, and the two do not scale
  together.
* **Nothing about recall.** Every arm is bit-identical by construction, which
  is a correctness claim, gated separately. If anything here moved recall that
  would be a defect, not a result.
* **No comparison against any other library.** faiss, hnswlib and the rest are
  absent on purpose: a fair cross-library comparison needs matched
  implementation effort on both sides, and nobody here has tuned faiss.
