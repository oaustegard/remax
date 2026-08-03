# Query-path speed

**This is a benchmark, not a gate.** There is no published constant for how
fast a Hamming scan should be, so none of these numbers can be gated: an
anchor is something your implementation did not produce, and wall-clock has
none. What *is* gated is the only claim with an anchor — that none of these
changes alters a single neighbour — by `bench/gates/query_path_gate.py`, which
proves itself by going red under 14 simulated defects.

Reproduce with `python3 bench/query_path_speed.py --trials 11`. Raw output in
`bench/results/query_path_speed.json`.

## The box

| | |
|---|---|
| CPU | Intel(R) Xeon(R) Processor @ 2.80 GHz, 4 cores / 4 threads, 1 socket |
| Cache | L1d 128 KiB (4x), L2 4 MiB (4x), L3 33 MiB (1 instance) |
| Memory | 16,461,176 kB (~15.7 GiB) |
| OS | Linux 6.18.5, x86_64, KVM guest |
| Python | 3.11.15 |
| NumPy | 2.4.4 |
| remax | 0.1.0 + this branch |
| Native kernel | available (gcc, `-O3 -mpopcnt`) |

Four cores and one socket. That is small, and it is the single biggest limit
on what the threading numbers mean — a ~2.7x on 4 cores says nothing about
what a 64-core socket does, in either direction, because the scan is
bandwidth-bound and core count and memory bandwidth do not scale together.

**Method.** Every number is the **minimum of 11 timed runs** after one warmup.
Minimum, not mean: the distribution is a hard floor (the work) plus a one-sided
noise tail (scheduling, page faults, co-tenants on a KVM guest). A mean
estimates the tail; a minimum estimates the floor, which is what the code
controls.

The box was otherwise idle. It was not, for the first run of this benchmark,
and that run reported 1.4-1.9x threading speedup where a quiet box reports
2.7-3.4x. Those numbers were discarded, not averaged in.

**Between-run variance is larger than the min-of-11 suggests, and matters more
than any single figure below.** Min-of-trials removes noise *within* a process;
it does nothing about drift *between* processes — thermal state, page
placement, co-tenants on a KVM guest. Three independent whole-benchmark runs
gave, for the same n=1e7 B=32 T=4 configuration:

| run | T=4 speedup | box state |
|---|---|---|
| 1 | 1.47x | contended (another job running) — discarded |
| 2 | 3.43x | quiet |
| 3 | 2.70x | quiet |

So two quiet runs disagreed by 27%. **Every table below is run 3**, the one
that produced the committed `query_path_speed.json`, so the markdown and the
JSON cannot drift apart. Read the speedups as "roughly 2.5-3.5x", not as three
significant figures — and if you are deciding something on a 20% difference,
re-run it yourself rather than trusting a number from this file.

## Matched implementation effort

A gate cannot check comparability. Two arms can each be individually correct
and still not be comparable, and nothing goes red — remax has already shipped
one benchmark that gave float32 a batched GEMM and remax a Python loop. So
each comparison below states what its arms share:

| Comparison | Why the arms are comparable |
|---|---|
| `threads=1` vs `threads=T` | Identical C kernel, identical `ctypes` call, identical output buffer. The T=1 arm *is* the T>1 arm with one block — same function, same code path. Neither side tuned. |
| unblocked vs blocked | Identical kernel, and the identical **number of popcount operations**: `n*m*B` either way. Only the loop order differs. The unblocked arm is `block=n` through the same function, not a reimplementation. |
| `argpartition` vs counting select | Both NumPy, both exact. The harness asserts on every trial that they return the **byte-identical permutation**, and that it equals `np.argsort(kind="stable")[:k]` — a speed comparison between a right answer and a wrong one is meaningless. The argpartition arm is the shipped implementation lifted out verbatim, not a strawman. |
| per-query vs batched tables | Both build **bit-identical** tables (asserted). One batched GEMM against m small ones. |
| `load` vs `mmap` | Same bytes, same kernel, same queries, same process shape. Only residency differs. |

**The one asymmetry, declared.** The counting select's histogram is *chunked*,
because unchunked `np.bincount` measured 58 ms against 19 ms at n=1e7 — it
casts its input to `intp` internally and the cast blows the cache. That is a
tuning choice made on one arm. `np.argpartition` has no analogous knob, so
effort *available* is not symmetric here. This note is the disclosure, not a
correction; there is nothing to correct, only something to say out loud.

---

## 1. Threaded scan

The C kernel is reached through `ctypes`, which releases the GIL, so row blocks
scan in parallel with no change to the C at all.

| n | B | index MB | threads | serial ms | threaded ms | speedup | GB/s |
|---|---|---|---|---|---|---|---|
| 100,000 | 32 | 3.2 | 2 | 0.351 | 0.334 | 1.05x | 9.6 |
| 100,000 | 32 | 3.2 | 4 | 0.351 | 0.355 | 0.99x | 9.0 |
| 100,000 | 32 | 3.2 | 8 | 0.351 | 0.347 | 1.01x | 9.2 |
| 1,000,000 | 32 | 32 | 2 | 5.35 | 3.25 | 1.65x | 9.9 |
| 1,000,000 | 32 | 32 | 4 | 5.35 | 1.97 | **2.72x** | 16.3 |
| 1,000,000 | 32 | 32 | 8 | 5.35 | 2.06 | 2.59x | 15.5 |
| 10,000,000 | 32 | 320 | 2 | 50.4 | 27.2 | 1.85x | 11.7 |
| 10,000,000 | 32 | 320 | 4 | 50.4 | 18.7 | **2.70x** | 17.1 |
| 10,000,000 | 32 | 320 | 8 | 50.4 | 17.8 | 2.83x | 17.9 |
| 1,000,000 | 96 | 96 | 2 | 14.1 | 7.75 | 1.81x | 12.4 |
| 1,000,000 | 96 | 96 | 4 | 14.1 | 5.14 | **2.73x** | 18.7 |
| 1,000,000 | 96 | 96 | 8 | 14.1 | 6.04 | 2.33x | 15.9 |

Every row was checked bit-identical to the serial scan.

The n=100,000 row is not a small speedup — it is **no threading at all**. At
3.2 MB the corpus is below `_MIN_BYTES_PER_THREAD` for a second worker, so
`hamming_distances` bypasses the pool and all three thread counts run the same
serial code. The 0.99-1.05x spread is noise around one implementation, and it
is a useful calibration of how much noise to expect elsewhere in this table.

That bypass is there because of a measurement, not a hunch. The threshold was
originally 16,384 **rows**, which at B=32 is 512 KB, and threading at that size
measured a real **slowdown**. Measured with the threshold temporarily removed
so the pool was actually reached (min of 15, separate from the three
whole-benchmark runs above):

| bytes/thread | T=4 speedup |
|---|---|
| 0.8 MB | 0.76x |
| 1.6 MB | 1.48x |
| 3.2 MB | 2.08x |
| 6.4 MB | 3.27x |

Pool dispatch costs ~60 µs for 2-4 tasks (~167 µs at 8, oversubscribed on 4
cores) against a ~9.7 GB/s single-core scan. Break-even is around 1 MB per
thread; the floor now sits at 2 MB, and it is expressed in bytes rather than
rows because the scan is bandwidth-bound and a row means different work at
B=32 and B=96.

**The correctness gate could not have found this.** The answers were right at
every size the whole time. A slowdown shipped as an optimisation is invisible
to an anchor.

### Scope

Measured at n = 1e5, 1e6, 1e7, at B = 32 and 96, at T = 1, 2, 4, 8, on 4 cores.
Nothing here supports a claim about a different core count, a different socket
count, or NUMA. The aggregate ceiling — ~18 GB/s in this run, ~24 GB/s in the
previous one — is this box's memory system, and it is the thing that stops T=8
from beating T=4 by much.

---

## 2. Counting select vs argpartition

Both exact, both returning the identical permutation, verified against
`np.argsort(kind="stable")[:k]` on every trial.

| n | k | argpartition ms | counting ms | speedup | argpartition alloc | counting alloc |
|---|---|---|---|---|---|---|
| 10,000 | 10 | 0.072 | 0.047 | 1.54x | 80 kB | 2.1 kB |
| 10,000 | 100 | 0.030 | 0.046 | 0.65x | 80 kB | 2.1 kB |
| 100,000 | 10 | 0.208 | 0.213 | 0.98x | 800 kB | 2.1 kB |
| 100,000 | 100 | 0.227 | 0.248 | 0.92x | 800 kB | 2.1 kB |
| 300,000 | 10 | 0.615 | 0.702 | 0.88x | 2.4 MB | 2.1 kB |
| 300,000 | 100 | 0.642 | 0.713 | 0.90x | 2.4 MB | 2.1 kB |
| 1,000,000 | 10 | 2.52 | 2.65 | 0.95x | 8 MB | 2.1 kB |
| 1,000,000 | 100 | 2.53 | 2.82 | 0.90x | 8 MB | 2.1 kB |
| 3,000,000 | 10 | 12.2 | 8.79 | 1.39x | 24 MB | 2.1 kB |
| 3,000,000 | 100 | 11.9 | 10.5 | 1.14x | 24 MB | 2.1 kB |
| 10,000,000 | 10 | 76.5 | 29.4 | **2.60x** | 80 MB | 2.1 kB |
| 10,000,000 | 100 | 77.5 | 31.0 | **2.50x** | 80 MB | 2.1 kB |

Distances are `binomial(8B, 0.5)` at B=32 — 257 possible values, tie-dense,
which is the shape a real Hamming scan produces.

The crossover is around **n = 2e6**, and `COUNTING_MIN_N` is set to 2^21 =
2,097,152 accordingly. An earlier 2^20 put the dispatch inside the losing
region; the benchmark is what moved it.

Two things worth separating:

* **Time.** The counting path wins only above the threshold, by up to 2.8x at
  n=1e7. Below it, argpartition wins by up to ~8%.
* **Allocation.** The counting path allocates ~2 kB at *every* size against
  argpartition's `8n` bytes — 80 MB at n=1e7, on a code path whose entire
  premise is not touching more memory than the index it scans. That advantage
  exists everywhere, including where the counting path loses on time.

The threshold is set on **time**, because remax has no measurement showing the
allocation matters to anyone. A caller who knows it matters to them can call
`counting_top_k` directly at any size.

### Scope

Measured at n = 1e4 to 1e7, k = 10 and 100, one alphabet (257 values, B=32),
one distribution (binomial). It says nothing about B=256 (2049 values), about
k comparable to n, or about a distribution with a very different tie structure
— a near-uniform alphabet moves the collection cost, which is the part the
histogram does not help with.

---

## 3. Blocked batch scan

The m-query loop read the whole corpus m times. It now reads a cache-sized
block once and scores it against all m queries. Same kernel, same `n*m*B`
popcount operations, different loop order.

Reported as a full **2x2** — {unblocked, blocked} x {1 thread, 4 threads} —
because "blocked and threaded vs unblocked and serial" is two changes in one
number, and it would credit blocking with the threading win. All four arms
verified bit-identical.

| n | B | m | block rows | unblocked t1 | unblocked t4 | blocked t1 | blocked t4 | blocking alone | threads alone | both | blocking *on top of* threads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e6 | 32 | 1 | (none) | 8.89 ms | 5.82 | 9.04 | 5.65 | 0.98x | 1.53x | 1.57x | 1.03x |
| 1e6 | 32 | 8 | 131,072 | 68.9 ms | 44.1 | 54.2 | 35.7 | 1.27x | 1.56x | 1.93x | 1.23x |
| 1e6 | 32 | 32 | 131,072 | 280 ms | 168 | 207 | 131 | 1.35x | 1.67x | **2.13x** | 1.28x |
| 1e7 | 32 | 1 | (none) | 78.6 ms | 42.5 | 78.0 | 43.3 | 1.01x | 1.85x | 1.82x | 0.98x |
| 1e7 | 32 | 8 | 131,072 | 647 ms | 354 | 553 | 357 | 1.17x | 1.83x | 1.81x | 0.99x |
| 1e7 | 32 | 32 | 131,072 | 2,585 ms | 1,438 | 2,095 | 1,337 | 1.23x | 1.80x | 1.93x | 1.08x |
| 1e6 | 96 | 1 | (none) | 18.7 ms | 7.85 | 18.3 | 7.98 | 1.02x | 2.38x | 2.34x | 0.98x |
| 1e6 | 96 | 8 | 43,690 | 147 ms | 66.6 | 105 | 64.7 | 1.40x | 2.20x | 2.27x | 1.03x |
| 1e6 | 96 | 32 | 43,690 | 588 ms | 289 | 368 | 226 | **1.60x** | 2.04x | **2.60x** | 1.27x |

**Blocking is the smaller of the two effects.** It buys 1.2-1.6x for a batch;
threading buys 1.5-2.4x on the same shapes. At m=1 blocking is correctly a
no-op (`_resolve_block` declines to block a single query) and the 0.98-1.02x
column confirms the bookkeeping costs nothing measurable.

Blocking and threading are **not** additive, and at n=1e7 they barely compose
at all. Blocking is worth 1.17-1.23x on its own there, and 0.99-1.08x once
four threads are already running — i.e. essentially nothing. Both attack the
same bottleneck from different directions: once the scan is spread over four
cores, the DRAM traffic blocking would have saved is no longer what limits it.

The gain is much smaller than the traffic arithmetic suggests. Reading the
corpus once instead of m times is a 6-30x reduction in bytes moved, and it
converts to 1.2-1.6x. Two reasons, both visible in the table:

* At n=1e6, B=32 the whole 32 MB index nearly fits the 33 MiB L3 already, so
  there was little DRAM traffic to save. B=96 (96 MB, comfortably past L3) is
  where blocking does best — 1.60x at m=32, against 1.35x for the same m at
  B=32.
* The blocked path pays more Python. At n=1e7 with 131,072-row blocks and
  m=32 that is 77 blocks x 32 queries = 2,464 `hamming_distances` calls plus
  2,464 selects and merges, where the unblocked path makes 32 of each. Some of
  the bandwidth win goes straight back out in interpreter overhead.

### Scope

Measured at n = 1e6 and 1e7, B = 32 and 96, m = 1, 8, 32, k = 10, with the
automatic block size (4 MB of codes). Block size itself was not swept, so
"4 MB is the right block" is a design choice justified by L2 size, **not** a
measured optimum. k was held at 10 throughout; a larger k makes the per-block
select and merge more expensive and would move these numbers down.

---

## 4. Asymmetric table hoist

`search_asymmetric` built its `(B, 256)` byte-value table once per query inside
its loop. It now builds all m in one batched GEMM before it. Bit-identical.

| d | m | per-query µs | batched µs | speedup |
|---|---|---|---|---|
| 256 | 1 | 8.0 | 7.1 | 1.12x |
| 256 | 8 | 70.4 | 38.5 | 1.83x |
| 256 | 32 | 291 | 154 | 1.89x |
| 768 | 1 | 14.9 | 14.3 | 1.05x |
| 768 | 8 | 116 | 72.3 | 1.61x |
| 768 | 32 | 471 | 402 | 1.17x |
| 2,048 | 1 | 38.2 | 39.0 | 0.98x |
| 2,048 | 8 | 331 | 303 | 1.09x |
| 2,048 | 32 | 1,350 | 1,063 | 1.27x |

**And it does not matter.** Put next to the scan it feeds, at n=200,000:

| d | table build µs | gather-and-sum µs | table share | asymmetric vs Hamming |
|---|---|---|---|---|
| 256 | 9.6 | 35,935 | **0.027%** | 45.1x |
| 768 | 12.4 | 98,387 | **0.013%** | 37.9x |

The table build is one to three parts in ten thousand of an asymmetric search.
Halving it is unobservable. The hoist is worth keeping because it is free and
bit-identical, not because anyone will notice it — and at d=2048, m=1 it is
marginally *negative* (0.98x), which is within noise but is certainly not a
win. The per-(d, m) speedups also move by 20-40% between runs, which is
another way of saying the same thing: nothing here is load-bearing.

The number in that table that actually matters is the last column:
**asymmetric scoring costs ~38-45x a Hamming scan** at these shapes, because
the gather-and-sum cannot use the popcount kernel. That is the whole reason
`Corpus.search(asymmetric=True)` is opt-in rather than the default, despite
being worth +0.019 nDCG@10 at 128 B/vector. Buying that recall costs roughly
40x the query time; whether that trade is right is a decision for the caller,
and it needs both numbers in front of it.

### Scope

Table build measured at d = 256, 768, 2048 and m = 1, 8, 32. The
share-of-total and asymmetric-vs-Hamming columns are at n = 200,000 only, and
the ratio moves with n — both sides are linear in n, so it should be roughly
stable, but "should be" is not a measurement and n=200,000 is what was run.

---

## 5. mmap vs load residency

n = 8,000,000, d = 256 (B = 32) — a **256 MB** index. Each arm runs in its own
subprocess with the page cache dropped first (`/proc/sys/vm/drop_caches`), so
"cold" is genuinely cold and RSS starts clean.

| residency | open ms | first query ms | warm query ms | RSS added at open | RSS at end |
|---|---|---|---|---|---|
| `load` | 391 | 91.6 | 65.8 | **250 MB** | 325 MB |
| `mmap` | 40.5 | 152 | 61.7 | **5.7 MB** | 325 MB |

Read this carefully, because the headline goes three ways at once:

* **Open is 9.7x faster** (40.5 ms vs 391 ms) and costs **44x less resident
  memory** — 5.7 MB against 250 MB. `np.fromfile` must read the whole 256 MB
  index before returning; `np.memmap` returns after establishing the mapping.
* **The first query is 1.7x slower** (152 ms vs 91.6 ms). The faults `load`
  paid up front, `mmap` pays on first touch, one at a time inside the scan
  rather than in one sequential read. The work did not disappear; it moved.
* **Warm queries are the same** (61.7 vs 65.8 ms — `mmap` nominally faster,
  which is noise, not an effect). Once the pages are resident it is the same
  scan over the same bytes.

Total cold path to first answer: **483 ms for `load`, 193 ms for `mmap`**. The
open-time ratio is the least stable number in this document — the previous run
measured 521 ms / 202 ms for the same configuration, a 2.6x rather than 9.7x,
because `load`'s open is dominated by a 256 MB cold read whose throughput
depends on the host's disk cache state in ways this benchmark does not
control. The *direction* is solid and the RSS figure is solid; the multiplier
is not.

**RSS at end is 325 MB for both, and that is not a wash.** Under `load` those
250 MB are private anonymous heap: not shareable, not evictable, and paid again
in full by every process that opens the index. Under `mmap` they are page-cache
pages backed by the file: shared between processes that map the same index, and
reclaimable under memory pressure. The RSS number is the same; what the
kernel can do with it is not. That distinction is the reason the option
exists, and this benchmark **does not measure it** — see below.

### Scope, and what this does not show

* One index size (256 MB), one dimension (d=256), one machine with 15.7 GiB of
  RAM. The index fits in RAM with room to spare, which is precisely the regime
  where mmap's advantage is smallest. The case it exists for — an index larger
  than RAM, or several processes sharing one — is **not measured here**.
* **The sharing benefit is not measured.** Only one process maps the index.
  That page-cache pages are shared across processes is a property of the
  kernel, not something this benchmark demonstrated.
* **The eviction benefit is not measured.** Nothing here applies memory
  pressure, so "evictable" is an argument from how `mmap` works, not an
  observation.
* The corpus was written directly (header + random bytes) with an **empty
  `meta.db`**, because encoding 8M real embeddings at d=256 needs 8 GB of
  float32 input and a populated metadata store would add tens of seconds of
  unrelated SQLite work to every configuration. Neither choice touches the
  quantity being compared — index residency — but both are stated because a
  reader should not have to guess.
* `open_ms` includes rebuilding the quantizer's `(256, 256)` Haar rotation,
  which is common to both arms. That is a constant added to both, so it
  *understates* the ratio between them.

---

## What none of this establishes

* **Nothing about a corpus larger than n = 1e7.** Sizes actually run are
  1e4-1e7 for select, 1e5-1e7 for scan, 1e6-1e7 for batch, and 8e6 for mmap.
  A bandwidth-bound scan changes regime at every cache boundary — this box
  crosses one between n=1e5 (L3-resident, 9.7 GB/s single-core) and n=1e6
  (6.7 GB/s) — so extrapolating to 1e8 is not conservative, it is unfounded.
* **Nothing about other hardware.** Four cores, one socket, ~24 GB/s aggregate.
  Threading speedups are bounded by the memory system, not by the core count,
  and the two do not scale together.
* **Nothing about recall.** These changes are bit-identical by construction,
  which is a correctness claim, gated separately, and is the reason recall does
  not appear in this document at all. If a change here altered recall, that
  would be a defect, not a result.
* **No comparison against any other library.** faiss, hnswlib and everything
  else are absent on purpose: a fair cross-library comparison needs matched
  implementation effort on both sides, and nobody here has tuned faiss.
