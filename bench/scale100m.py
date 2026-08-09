#!/usr/bin/env python3
"""What a single query costs at n=1e8. Issue #70.

``bench/results/QUERY_PATH_SPEED.md`` closes by saying it establishes
**nothing about a corpus larger than n = 1e7**, and means it: a bandwidth-bound
scan changes regime at every cache boundary, so extrapolating is not
conservative. This file measures the sizes that document declined to.

The question it exists for
--------------------------
The blog post *Three Gigs to Search a Hundred Million Papers* claims a
single-threaded brute-force top-100 in "well under a second" at n=1e8. That is
now measurable rather than extrapolated — 100M x 32 B is 3.2 GB and fits in a
16 GB container. Issue #70 measured a standalone C kernel and found the *scan*
comfortably inside the budget while *selection* blew it, at roughly 2x the
scan, and asked three things of remax's own path, which is not that kernel:

1. what does ``hamming_distances`` + ``stable_top_k`` actually cost end to end;
2. does the blocked path already close the gap for one query;
3. is a threshold-fused variant worth carrying.

Arms, and why they are comparable
---------------------------------
Every arm below scans the identical corpus with the identical C kernel and
performs the identical ``n * B`` popcount operations. They differ only in what
happens to the distances afterwards, and every one of them is checked against
``np.argsort(kind="stable")[:k]`` on every trial — a speed comparison between a
right answer and a wrong one measures nothing.

============================  ==============================================
Arm                           What it is
============================  ==============================================
scan only                     ``hamming_distances`` into a reused buffer.
                              The floor: no selection at all, so no arm
                              below can beat it.
scan + stable_top_k           The shipped unblocked path, which is what
                              ``hamming_topk_batch`` ran for m=1 before this
                              issue. Two sub-arms, fresh ``(n,) int32``
                              output vs a reused ``out=`` buffer, because at
                              n=1e8 that allocation is 400 MB.
scan + argpartition           The same scan with the selection remax
                              *avoids* above ``COUNTING_MIN_N``. Present
                              because it is the arm issue #70 measured, and
                              the difference between the two is most of the
                              disagreement.
blocked, per-block top-k      The pre-#70 blocked path, reached with an
                              explicit ``block=``. Question 2.
blocked, running threshold    The shipped blocked path. Question 3.
============================  ==============================================

Method is ``bench/query_path_speed.py``'s: minimum of ``--trials`` runs after a
warmup, a stated box, and stated scope. Minimum rather than mean because the
distribution is a hard floor plus a one-sided noise tail.

**Warm, and it matters more here than anywhere else in this repo.** Issue #70
first measured argpartition at 5391 ms and warm at 714 ms — 7.6x apart, the
difference being page faults on a freshly written 400 MB score array plus an
800 MB permutation allocation. A cold number would make selection look
catastrophic rather than merely dominant. Everything here is warm and says so.

Usage
-----
::

    python3 bench/scale100m.py                        # 1e6 .. 1e8
    python3 bench/scale100m.py --sizes 1e6 1e7        # quicker
    python3 bench/scale100m.py --only blocks          # block-size sweep
    python3 bench/scale100m.py --json out.json

n=1e8 needs ~4 GB of RAM (3.2 GB of codes, 400 MB of scores, and argpartition's
800 MB permutation on the one arm that allocates it) and takes a few minutes.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import remax  # noqa: E402
import remax.packing as packing  # noqa: E402
from remax.packing import (  # noqa: E402
    hamming_distances,
    hamming_topk_batch,
    stable_top_k,
)

SECTIONS = ("end2end", "blocks", "select", "chunk", "batch")

#: The batch section is quadratic in a way the others are not (m full scans of
#: n rows), so it runs at the smaller sizes only. m=32 at n=1e8 is ~15 s per
#: trial, which buys nothing the n=1e7 row does not already show.
BATCH_MAX_N = 10**7
DEFAULT_SIZES = (10**6, 10**7, 10**8)


# ── harness ──────────────────────────────────────────────────────────────────

def best_of(fn, trials: int, warmup: int = 1):
    """Minimum wall-clock over ``trials`` runs, in seconds, plus the result."""
    for _ in range(warmup):
        r = fn()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        best = min(best, time.perf_counter() - t0)
    return best, r


def box() -> dict:
    info = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpus_available": packing._cpu_count(),
        "native_kernel": remax.NATIVE_AVAILABLE,
        "remax": remax.__version__,
    }
    for path, key, prefix in (
        ("/proc/cpuinfo", "cpu", "model name"),
        ("/proc/meminfo", "memtotal", "MemTotal"),
    ):
        try:
            for line in Path(path).read_text().splitlines():
                if line.startswith(prefix):
                    info[key] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return info


def rand_codes(n: int, b: int, seed: int = 0) -> np.ndarray:
    """Uniform random bytes, written in chunks so the generator never holds a
    second copy of a multi-GB array."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, b), dtype=np.uint8)
    step = max(1, 8_000_000 // b)
    for s in range(0, n, step):
        e = min(s + step, n)
        out[s:e] = rng.integers(0, 256, size=(e - s, b), dtype=np.uint8)
    return out


def table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(no rows)"
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.1f}" if abs(v) >= 100 else f"{v:,.3f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def argpartition_arm(d: np.ndarray, k: int) -> np.ndarray:
    """``stable_top_k``'s comparison branch, lifted out verbatim.

    Not a strawman: this is the shipped code below ``COUNTING_MIN_N``, and it
    is what the counting select replaced above it. Widening included, because
    a bare ``argpartition`` would be answering a different (wrong) question.
    """
    part = np.argpartition(d, k - 1)
    pivot = d[part[k - 1]]
    cand = np.flatnonzero(d <= pivot)
    return cand[np.argsort(d[cand], kind="stable")][:k]


# ── 1. end to end, single query ──────────────────────────────────────────────

def bench_end2end(sizes, b, k, trials, threads) -> list[dict]:
    rows = []
    for n in sizes:
        codes = rand_codes(n, b, seed=n % 97)
        q = rand_codes(1, b, seed=1)[0]
        qq = q[None, :]
        bound = 8 * b
        scratch = np.empty(n, dtype=np.int32)
        hamming_distances(codes, q, out=scratch, threads=1)
        ref = np.argsort(scratch, kind="stable")[:k]

        def checked(fn):
            t, r = best_of(fn, trials)
            idx = r[0][0] if isinstance(r, tuple) else r
            return t, bool(np.array_equal(np.asarray(idx).ravel()[:k], ref))

        t_scan, _ = best_of(
            lambda: hamming_distances(codes, q, out=scratch, threads=1), trials
        )
        t_scan_t, _ = best_of(
            lambda: hamming_distances(codes, q, out=scratch, threads=threads),
            trials,
        )

        def unblocked_alloc():
            d = hamming_distances(codes, q, threads=1)
            return stable_top_k(d, k, value_bound=bound)
        def unblocked_reuse():
            hamming_distances(codes, q, out=scratch, threads=1)
            return stable_top_k(scratch, k, value_bound=bound)
        def unblocked_argpart():
            hamming_distances(codes, q, out=scratch, threads=1)
            return argpartition_arm(scratch, k)
        def fused():
            return hamming_topk_batch(codes, qq, k, threads=1)
        def fused_t():
            return hamming_topk_batch(codes, qq, k, threads=threads)
        def unblocked_t():
            hamming_distances(codes, q, out=scratch, threads=threads)
            return stable_top_k(scratch, k, value_bound=bound)

        t_alloc, ok_a = checked(unblocked_alloc)
        t_reuse, ok_r = checked(unblocked_reuse)
        t_ap, ok_p = checked(unblocked_argpart)
        t_fused, ok_f = checked(fused)
        t_unb_t, ok_ut = checked(unblocked_t)
        t_fused_t, ok_ft = checked(fused_t)

        rows.append({
            "n": n, "B": b, "k": k,
            "scan_t1_ms": t_scan * 1e3,
            "scan_gb_s": codes.nbytes / t_scan / 1e9,
            "unblocked_alloc_ms": t_alloc * 1e3,
            "unblocked_reuse_ms": t_reuse * 1e3,
            "unblocked_argpart_ms": t_ap * 1e3,
            "blocked_ms": t_fused * 1e3,
            "select_share_unblocked": (t_alloc - t_scan) / t_scan,
            "select_share_blocked": (t_fused - t_scan) / t_scan,
            "speedup": t_alloc / t_fused,
            f"scan_t{threads}_ms": t_scan_t * 1e3,
            f"unblocked_t{threads}_ms": t_unb_t * 1e3,
            f"blocked_t{threads}_ms": t_fused_t * 1e3,
            f"speedup_t{threads}": t_unb_t / t_fused_t,
            "exact": all((ok_a, ok_r, ok_p, ok_f, ok_ut, ok_ft)),
        })
        del codes, scratch
    return rows


# ── 2. block size, and per-block top-k vs running threshold ─────────────────

def bench_blocks(sizes, b, k, trials) -> list[dict]:
    """Question 2 against question 3, at matched block sizes.

    ``per_block_topk_ms`` is the pre-#70 blocked path: same blocking, same
    merge, but every block ranked in full by ``stable_top_k`` instead of
    filtered against the running k-th distance. Reconstructed here rather than
    imported because the shipped code no longer contains it — the two arms
    share the scan, the block boundaries and the merge, and differ in one step.
    """
    rows = []
    for n in sizes:
        codes = rand_codes(n, b, seed=n % 97)
        q = rand_codes(1, b, seed=1)[0]
        qq = q[None, :]
        bound = 8 * b
        d_ref = hamming_distances(codes, q, threads=1)
        ref = np.argsort(d_ref, kind="stable")[:k]
        del d_ref

        def per_block_topk(blk):
            buf = np.empty(min(blk, n), dtype=np.int32)
            run_i = np.empty(0, dtype=np.intp)
            run_d = np.empty(0, dtype=np.int32)
            for start in range(0, n, blk):
                stop = min(start + blk, n)
                dd = buf[: stop - start]
                hamming_distances(codes[start:stop], q, out=dd, threads=1)
                order = stable_top_k(dd, min(k, dd.size), value_bound=bound)
                run_i, run_d = packing._merge_topk(
                    run_i, run_d,
                    order.astype(np.intp, copy=False) + start, dd[order], k,
                )
            return run_i

        auto_blk = packing._resolve_block(None, n, 1, b)
        for blk in sorted({1 << 15, 1 << 17, 1 << 19, 1 << 20, 1 << 21,
                            1 << 23, auto_blk}):
            if blk > n:
                continue
            t_old, r_old = best_of(lambda blk=blk: per_block_topk(blk), trials)
            t_new, r_new = best_of(
                lambda blk=blk: hamming_topk_batch(codes, qq, k, block=blk),
                trials,
            )
            rows.append({
                "n": n, "block_rows": blk,
                "score_buf_mib": blk * 4 / 2**20,
                "code_buf_mib": blk * b / 2**20,
                "per_block_topk_ms": t_old * 1e3,
                "running_threshold_ms": t_new * 1e3,
                "speedup": t_old / t_new,
                "is_auto_block": blk == auto_blk,
                "exact": bool(np.array_equal(r_old, ref)
                              and np.array_equal(r_new[0][0], ref)),
            })
        del codes
    return rows


# ── 3. selection alone, on a materialised score array ────────────────────────

def bench_select(sizes, b, k, trials) -> list[dict]:
    """Where the unblocked path's time goes once the scan is paid for."""
    rows = []
    for n in sizes:
        codes = rand_codes(n, b, seed=n % 97)
        q = rand_codes(1, b, seed=1)[0]
        d = hamming_distances(codes, q, threads=1)
        del codes
        bound = 8 * b
        ref = np.argsort(d, kind="stable")[:k]
        t_count, r_count = best_of(
            lambda: packing._counting_top_k(d, k, bound), trials
        )
        t_hist, _ = best_of(lambda: packing._histogram(d, bound), trials)
        t_ap, r_ap = best_of(lambda: argpartition_arm(d, k), trials)
        rows.append({
            "n": n, "k": k,
            "counting_ms": t_count * 1e3,
            "histogram_pass_ms": t_hist * 1e3,
            "histogram_share": t_hist / t_count,
            "argpartition_ms": t_ap * 1e3,
            "counting_vs_argpartition": t_ap / t_count,
            "argpartition_alloc_mb": n * np.dtype(np.intp).itemsize / 1e6,
            "exact": bool(np.array_equal(r_count, ref)
                          and np.array_equal(r_ap, ref)),
        })
        del d
    return rows


# ── 4. _COUNT_CHUNK, re-swept at these sizes ─────────────────────────────────

def bench_chunk(sizes, b, k, trials) -> list[dict]:
    """The counting select's chunk size, at sizes it was never set on.

    ``_COUNT_CHUNK`` is 2^20, chosen at n=1e7 where the alternatives measured
    were 2^18 to 2^20 (``bench/results/QUERY_PATH_SPEED.md``, section 2). The
    histogram pass is the majority of selection at n=1e8, so the constant is
    worth re-asking there. This section only measures; nothing dispatches on it.
    """
    rows = []
    orig = packing._COUNT_CHUNK
    try:
        for n in sizes:
            codes = rand_codes(n, b, seed=n % 97)
            q = rand_codes(1, b, seed=1)[0]
            d = hamming_distances(codes, q, threads=1)
            del codes
            bound = 8 * b
            ref = np.argsort(d, kind="stable")[:k]
            row = {"n": n, "k": k, "shipped_chunk": orig, "exact": True}
            for c in (1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22):
                packing._COUNT_CHUNK = c
                t, r = best_of(
                    lambda: packing._counting_top_k(d, k, bound), trials
                )
                row[f"chunk_{c}_ms"] = t * 1e3
                row["exact"] &= bool(np.array_equal(r, ref))
            packing._COUNT_CHUNK = orig
            row["best_vs_shipped"] = (
                row[f"chunk_{orig}_ms"]
                / min(v for key, v in row.items()
                      if isinstance(key, str) and key.startswith("chunk_"))
            )
            rows.append(row)
            del d
    finally:
        packing._COUNT_CHUNK = orig
    return rows


# ── 5. the m > 1 batch path, which the same filter also changed ──────────────

def bench_batch(sizes, b, k, trials, threads) -> list[dict]:
    """A regression check with a number attached.

    The running threshold was added for the single-query case, but it replaced
    the per-block ``stable_top_k`` for *every* m, and a batch is the shape that
    path was originally written for. This is here so the change is not shipped
    on an untested claim of "should be fine".

    Both arms use the same automatic block size, the same merge and the same
    scan; only the per-block step differs, exactly as in section 2.
    """
    rows = []
    for n in [s for s in sizes if s <= BATCH_MAX_N]:
        codes = rand_codes(n, b, seed=n % 97)
        for m in (2, 8, 32):
            q_codes = rand_codes(m, b, seed=m)
            blk = packing._resolve_block(None, n, m, b)
            for t in (1, threads):
                t_old, r_old = best_of(
                    lambda: _per_block_topk_batch(codes, q_codes, k, blk, t),
                    trials,
                )
                t_new, r_new = best_of(
                    lambda: hamming_topk_batch(codes, q_codes, k, threads=t),
                    trials,
                )
                rows.append({
                    "n": n, "m": m, "threads": t, "block_rows": blk,
                    "per_block_topk_ms": t_old * 1e3,
                    "running_threshold_ms": t_new * 1e3,
                    "speedup": t_old / t_new,
                    "identical": bool(np.array_equal(r_old, r_new[0])),
                })
        del codes
    return rows


def _per_block_topk_batch(codes, q_codes, k, blk, threads):
    """The pre-#70 blocked batch loop, reconstructed. Section 2's arm, for m>1."""
    n, code_bytes = codes.shape
    m = q_codes.shape[0]
    bound = 8 * code_bytes
    buf = np.empty((m, blk), dtype=np.int32)
    run_i = [np.empty(0, dtype=np.intp) for _ in range(m)]
    run_d = [np.empty(0, dtype=np.int32) for _ in range(m)]
    workers = packing.resolve_threads(threads)
    pool = packing._pool(workers) if workers > 1 and m > 1 else None
    for start in range(0, n, blk):
        stop = min(start + blk, n)
        length = stop - start
        sub = codes[start:stop]
        if pool is None:
            for i in range(m):
                hamming_distances(sub, q_codes[i], out=buf[i, :length],
                                  threads=1)
        else:
            for f in [pool.submit(hamming_distances, sub, q_codes[i],
                                  out=buf[i, :length], threads=1)
                      for i in range(m)]:
                f.result()
        for i in range(m):
            d = buf[i, :length]
            order = stable_top_k(d, min(k, length), value_bound=bound)
            run_i[i], run_d[i] = packing._merge_topk(
                run_i[i], run_d[i],
                order.astype(np.intp, copy=False) + start, d[order], k,
            )
    return np.array(run_i)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", nargs="+", type=float, default=DEFAULT_SIZES,
                    help="corpus sizes (accepts 1e8)")
    ap.add_argument("--b", type=int, default=32, help="bytes per code")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=SECTIONS)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes]
    info = box()
    print("# remax single-query cost at scale (issue #70)\n")
    print("## The box\n")
    for key, val in info.items():
        print(f"- **{key}**: {val}")
    print(f"\nMinimum of {args.trials} runs after a warmup. B={args.b}, "
          f"k={args.k}. Every arm verified against "
          f"`np.argsort(kind='stable')[:k]`.\n")

    results = {"box": info, "trials": args.trials, "B": args.b, "k": args.k}
    t = args.threads

    if "end2end" in args.only:
        print("## 1. End to end, one query\n")
        rows = bench_end2end(sizes, args.b, args.k, args.trials, t)
        results["end2end"] = rows
        print(table(rows, ["n", "scan_t1_ms", "scan_gb_s",
                           "unblocked_alloc_ms", "unblocked_reuse_ms",
                           "unblocked_argpart_ms", "blocked_ms",
                           "select_share_unblocked", "select_share_blocked",
                           "speedup", "exact"]))
        print()
        print(table(rows, ["n", f"scan_t{t}_ms", f"unblocked_t{t}_ms",
                           f"blocked_t{t}_ms", f"speedup_t{t}", "exact"]))
        print()

    if "blocks" in args.only:
        print("## 2. Per-block top-k vs the running threshold\n")
        rows = bench_blocks(sizes, args.b, args.k, args.trials)
        results["blocks"] = rows
        print(table(rows, ["n", "block_rows", "score_buf_mib", "code_buf_mib",
                           "per_block_topk_ms", "running_threshold_ms",
                           "speedup", "is_auto_block", "exact"]))
        print()

    if "select" in args.only:
        print("## 3. Selection alone, on a materialised score array\n")
        rows = bench_select(sizes, args.b, args.k, args.trials)
        results["select"] = rows
        print(table(rows, ["n", "k", "counting_ms", "histogram_pass_ms",
                           "histogram_share", "argpartition_ms",
                           "counting_vs_argpartition",
                           "argpartition_alloc_mb", "exact"]))
        print()

    if "chunk" in args.only:
        print("## 4. _COUNT_CHUNK re-swept (measurement only, nothing "
              "dispatches on it)\n")
        rows = bench_chunk(sizes, args.b, args.k, args.trials)
        results["chunk"] = rows
        chunk_cols = ["n", "shipped_chunk"] + [
            f"chunk_{1 << c}_ms" for c in (14, 16, 18, 20, 22)
        ] + ["best_vs_shipped", "exact"]
        print(table(rows, chunk_cols))
        print()

    if "batch" in args.only:
        print(f"## 5. The m>1 batch path, same change (n <= "
              f"{BATCH_MAX_N:,})\n")
        rows = bench_batch(sizes, args.b, args.k, args.trials, t)
        results["batch"] = rows
        print(table(rows, ["n", "m", "threads", "block_rows",
                           "per_block_topk_ms", "running_threshold_ms",
                           "speedup", "identical"]))
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
