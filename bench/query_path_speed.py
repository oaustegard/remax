#!/usr/bin/env python3
"""Benchmark the query-path optimisations. Not a gate — a measurement.

There is no published constant for how fast a Hamming scan should be, so none
of this belongs in ``bench/gates/query_path_gate.py``. That gate checks the
only thing that has an anchor: **the answers do not change.** This file
measures the thing that does not: how long they take.

What that obliges, and what this file does about it
---------------------------------------------------
**Matched implementation effort.** A gate cannot check comparability — two
arms can each be individually correct and still not be comparable, and nothing
goes red. remax has already shipped one benchmark that gave float32 a batched
GEMM and remax a Python loop. So every comparison here says what the two arms
share:

===================  =========================================================
Comparison           Why the arms are comparable
===================  =========================================================
threads 1 vs T       Identical C kernel, identical ctypes call, identical
                     output buffer. The only difference is how many threads
                     call it. Neither side was tuned.
unblocked vs         Identical kernel, and the identical *number of popcount
blocked              operations* — n*m*B either way. Only the loop order
                     differs.
argpartition vs      Both NumPy, both exact, and the harness asserts on every
counting select      trial that they return the byte-identical permutation.
                     The argpartition arm is the shipped implementation, not
                     a strawman written for this file. See the caveat below.
per-query vs         Both build the identical tables (asserted bit-identical).
batched tables       One batched GEMM against m small ones.
load vs mmap         Same bytes, same kernel, same queries. Only residency
                     differs.
===================  =========================================================

The one asymmetry worth declaring: the counting select's histogram is
**chunked**, because unchunked ``np.bincount`` measured 58 ms against 19 ms at
n=1e7 (it casts to intp internally, and the cast blows the cache). That is a
tuning choice made on the counting arm. ``np.argpartition`` has no analogous
knob to tune, so the comparison is not symmetric in effort available. It
favours the counting arm, and this note is the disclosure rather than a
correction — there is nothing to correct, only something to say.

**Min of trials.** Every number is the minimum of ``--trials`` runs after
warmup. Minimum, not mean: the distribution is a floor plus scheduler noise,
so the mean measures the noise and the minimum measures the code.

**A stated box.** Printed in the header of every run and reproduced in
``bench/results/QUERY_PATH_SPEED.md``. These numbers are hardware-specific.
Re-measure before quoting them anywhere.

**Stated scope.** The sizes actually run are printed. Nothing here supports a
claim about a size that was not measured; a bandwidth-bound scan changes
regime at every cache boundary, so extrapolating an n=1e4 result to n=1e8 is
not conservative, it is wrong.

Usage
-----
::

    python3 bench/query_path_speed.py                 # everything
    python3 bench/query_path_speed.py --only scan select
    python3 bench/query_path_speed.py --trials 9 --json out.json

``--only mmap`` re-executes itself in a subprocess per configuration so each
gets a clean RSS and, where permitted, a dropped page cache.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import remax  # noqa: E402
import remax.packing as packing  # noqa: E402
from remax.packing import (  # noqa: E402
    asymmetric_scores,
    asymmetric_tables,
    counting_top_k,
    hamming_distances,
    hamming_topk_batch,
    scores_from_table,
)

SECTIONS = ("scan", "select", "batch", "asym", "mmap")


# ── harness ──────────────────────────────────────────────────────────────────

def best_of(fn, trials: int, warmup: int = 1) -> float:
    """Minimum wall-clock over ``trials`` runs, in seconds.

    The minimum rather than the mean: the timing distribution is a hard floor
    (the work) plus a one-sided noise tail (scheduling, page faults, other
    tenants). A mean over that estimates the tail; the minimum estimates the
    floor, which is the thing the code controls.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def box() -> dict:
    """Everything needed to know these numbers do not transfer."""
    info = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpus_available": packing._cpu_count(),
        "native_kernel": remax.NATIVE_AVAILABLE,
        "remax": remax.__version__,
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["memtotal"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return info


def rand_codes(n: int, b: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, b), dtype=np.uint8)


# ── 1. threaded scan ─────────────────────────────────────────────────────────

def bench_scan(trials: int) -> list[dict]:
    """Threads 1 vs T on the identical kernel.

    Both arms call ``_native.hamming_distances_native`` on a slice of the same
    array into a slice of the same output buffer. There is no second
    implementation to get wrong: the T=1 arm is the T>1 arm with one block.
    """
    rows = []
    for n, b in ((10**5, 32), (10**6, 32), (10**7, 32), (10**6, 96)):
        codes = rand_codes(n, b, seed=n % 97)
        query = rand_codes(1, b, seed=1)[0]
        out = np.empty(n, dtype=np.int32)
        ref = hamming_distances(codes, query, threads=1).copy()
        base = best_of(
            lambda: hamming_distances(codes, query, out=out, threads=1), trials
        )
        for t in (2, 4, 8):
            el = best_of(
                lambda t=t: hamming_distances(
                    codes, query, out=out, threads=t
                ),
                trials,
            )
            identical = bool(
                np.array_equal(
                    hamming_distances(codes, query, threads=t), ref
                )
            )
            rows.append({
                "n": n, "B": b, "threads": t,
                "serial_ms": base * 1e3, "ms": el * 1e3,
                "speedup": base / el,
                "gb_s": codes.nbytes / el / 1e9,
                "bit_identical": identical,
            })
        del codes, out
    return rows


# ── 2. counting select vs argpartition ───────────────────────────────────────

def _argpartition_top_k(dists, k):
    """The shipped comparison path, called directly.

    Lifted out so the counting path's dispatch cannot accidentally be timed on
    both sides of the comparison. It is the exact body of ``stable_top_k``'s
    argpartition branch — not a simplified stand-in.
    """
    n = dists.shape[0]
    k_eff = min(k, n)
    part = np.argpartition(dists, k_eff - 1)
    pivot = dists[part[k_eff - 1]]
    cand = np.flatnonzero(dists <= pivot)
    return cand[np.argsort(dists[cand], kind="stable")][:k_eff]


def bench_select(trials: int) -> list[dict]:
    rows = []
    for n in (10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7):
        for b in (32,):
            rng = np.random.default_rng(7)
            dists = rng.binomial(8 * b, 0.5, size=n).astype(np.int32)
            for k in (10, 100):
                a = _argpartition_top_k(dists, k)
                c = counting_top_k(dists, k, value_bound=8 * b)
                ref = np.argsort(dists, kind="stable")[:k]
                assert np.array_equal(a, ref) and np.array_equal(c, ref), (
                    "the arms disagree — a speed comparison between a right "
                    "answer and a wrong one is meaningless"
                )
                t_a = best_of(lambda: _argpartition_top_k(dists, k), trials)
                t_c = best_of(
                    lambda: counting_top_k(dists, k, value_bound=8 * b), trials
                )
                rows.append({
                    "n": n, "B": b, "k": k,
                    "argpartition_ms": t_a * 1e3,
                    "counting_ms": t_c * 1e3,
                    "speedup": t_a / t_c,
                    "identical": True,
                    "argpartition_alloc_bytes": 8 * n,
                    "counting_alloc_bytes": 8 * (8 * b + 1),
                })
    return rows


# ── 3. blocked batch scan ────────────────────────────────────────────────────

def bench_batch(trials: int) -> list[dict]:
    """Same kernel, same popcount count, different loop order.

    The unblocked arm is ``block=n``, which is literally the per-query loop
    this replaced — same function, same code path up to the branch. Nothing
    was reimplemented for the comparison.
    """
    rows = []
    for n, b in ((10**6, 32), (10**7, 32), (10**6, 96)):
        codes = rand_codes(n, b, seed=(n + b) % 89)
        for m in (1, 8, 32):
            q_codes = rand_codes(m, b, seed=m)
            # Full 2x2. Reporting only "blocked+threaded vs unblocked+serial"
            # would credit blocking with the threading win and vice versa —
            # two changes, one number, and no way to tell which did the work.
            arms = {
                ("unblocked", 1): dict(block=n, threads=1),
                ("unblocked", 4): dict(block=n, threads=4),
                ("blocked", 1): dict(block=None, threads=1),
                ("blocked", 4): dict(block=None, threads=4),
            }
            ref_i, ref_d = hamming_topk_batch(codes, q_codes, 10, block=n)
            timings = {}
            identical = True
            for key, kw in arms.items():
                timings[key] = best_of(
                    lambda kw=kw: hamming_topk_batch(
                        codes, q_codes, 10, **kw
                    ),
                    trials,
                )
                got_i, got_d = hamming_topk_batch(codes, q_codes, 10, **kw)
                identical &= bool(
                    np.array_equal(ref_i, got_i) and np.array_equal(ref_d, got_d)
                )
            base = timings[("unblocked", 1)]
            rows.append({
                "n": n, "B": b, "m": m,
                "block_rows": packing._resolve_block(None, n, m, b),
                "unblocked_t1_ms": base * 1e3,
                "unblocked_t4_ms": timings[("unblocked", 4)] * 1e3,
                "blocked_t1_ms": timings[("blocked", 1)] * 1e3,
                "blocked_t4_ms": timings[("blocked", 4)] * 1e3,
                "block_only": base / timings[("blocked", 1)],
                "threads_only": base / timings[("unblocked", 4)],
                "both": base / timings[("blocked", 4)],
                "block_on_top_of_threads":
                    timings[("unblocked", 4)] / timings[("blocked", 4)],
                "bit_identical": identical,
            })
        del codes
    return rows


# ── 4. asymmetric table hoist ────────────────────────────────────────────────

def bench_asym(trials: int) -> list[dict]:
    """m small GEMMs against one batched GEMM, building identical tables."""
    rows = []
    for d in (256, 768, 2048):
        nb = d // 8
        rng = np.random.default_rng(3)
        Q = rng.standard_normal((32, d)).astype(np.float32)
        for m in (1, 8, 32):
            q = np.ascontiguousarray(Q[:m])

            def per_query(q=q, nb=nb, m=m):
                for i in range(m):
                    asymmetric_tables(q[i], nb)

            def batched(q=q, nb=nb):
                asymmetric_tables(q, nb)

            t_p = best_of(per_query, trials)
            t_b = best_of(batched, trials)
            tabs, _ = asymmetric_tables(q, nb)
            same = all(
                np.array_equal(tabs[i], asymmetric_tables(q[i], nb)[0][0])
                for i in range(m)
            )
            rows.append({
                "d": d, "m": m,
                "per_query_us": t_p * 1e6,
                "batched_us": t_b * 1e6,
                "speedup": t_p / t_b,
                "per_query_each_us": t_p * 1e6 / m,
                "bit_identical": bool(same),
            })
    # Whole-search cost, so the table build is seen in proportion.
    for d in (256, 768):
        n = 200_000
        rng = np.random.default_rng(4)
        codes = rand_codes(n, d // 8, seed=d)
        q = rng.standard_normal(d).astype(np.float32)
        tables, offsets = asymmetric_tables(q, d // 8)
        t_scan = best_of(
            lambda: scores_from_table(tables[0], codes, float(offsets[0])),
            trials,
        )
        t_tab = best_of(lambda: asymmetric_tables(q, d // 8), trials)
        t_full = best_of(lambda: asymmetric_scores(q, codes), trials)
        t_ham = best_of(
            lambda: hamming_distances(codes, rand_codes(1, d // 8)[0]), trials
        )
        rows.append({
            "d": d, "n": n, "kind": "proportion",
            "table_us": t_tab * 1e6,
            "scan_us": t_scan * 1e6,
            "asym_total_us": t_full * 1e6,
            "hamming_us": t_ham * 1e6,
            "table_share_pct": 100.0 * t_tab / t_full,
            "asym_vs_hamming": t_full / t_ham,
        })
        del codes
    return rows


# ── 5. mmap residency (subprocess per configuration) ─────────────────────────

def _make_corpus(path: Path, n: int, d: int) -> None:
    """Write a corpus directly: header + random codes + an EMPTY meta.db.

    Codes are random bytes rather than encoded embeddings because encoding
    n=8e6 at d=256 would need 8 GB of float32 input, and the scan does not care
    what the bytes mean. The metadata store is left empty on purpose: this
    section measures index residency, and a populated meta.db would add tens of
    seconds of unrelated SQLite work to every configuration. Neither choice
    affects the quantity being compared, and both are stated in the writeup.
    """
    path.mkdir(parents=True, exist_ok=True)
    b = d // 8
    rng = np.random.default_rng(99)
    with open(path / "index.bin", "wb") as f:
        f.write(remax.corpus._write_header(n, d, 0))
        chunk = 1 << 20
        written = 0
        while written < n:
            rows = min(chunk, n - written)
            f.write(rng.integers(0, 256, size=(rows, b), dtype=np.uint8).tobytes())
            written += rows
    con = sqlite3.connect(str(path / "meta.db"))
    con.executescript(remax.corpus._SCHEMA)
    con.commit()
    con.close()
    (path / "rotation.json").write_text('{"rotation": "haar"}\n')


def _rss_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1])
    return -1


def _drop_caches() -> bool:
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except OSError:
        return False


def _mmap_child(path: str, residency: str, k: int, trials: int) -> None:
    """One measurement, in its own process: clean RSS, cold page cache."""
    cold = _drop_caches()
    rss0 = _rss_kb()
    t0 = time.perf_counter()
    c = remax.Corpus(Path(path), residency=residency)
    open_s = time.perf_counter() - t0
    rss_after_open = _rss_kb()

    rng = np.random.default_rng(5)
    q = rng.standard_normal(c.d).astype(np.float32)
    t0 = time.perf_counter()
    c.search(q, k=k)
    first_s = time.perf_counter() - t0
    warm_s = best_of(lambda: c.search(q, k=k), trials)
    rss_end = _rss_kb()
    c.close()
    print(json.dumps({
        "residency": residency, "n": c.n, "d": c.d,
        "cold_cache": cold,
        "open_ms": open_s * 1e3,
        "first_query_ms": first_s * 1e3,
        "warm_query_ms": warm_s * 1e3,
        "rss_start_mb": rss0 / 1024,
        "rss_after_open_mb": rss_after_open / 1024,
        "rss_end_mb": rss_end / 1024,
        "rss_open_delta_mb": (rss_after_open - rss0) / 1024,
    }))


def bench_mmap(trials: int, n: int, d: int) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="remax_mmap_bench_") as tmp:
        path = Path(tmp) / "corpus"
        _make_corpus(path, n, d)
        for residency in ("load", "mmap"):
            proc = subprocess.run(
                [sys.executable, __file__, "--mmap-child", str(path),
                 residency, "--trials", str(trials)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:  # pragma: no cover
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(f"mmap child failed for {residency}")
            rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return rows


# ── reporting ────────────────────────────────────────────────────────────────

def table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    out = [head, rule]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmap-child", nargs=2, metavar=("PATH", "RESIDENCY"))
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--mmap-n", type=int, default=8_000_000)
    ap.add_argument("--mmap-d", type=int, default=256)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    if args.mmap_child:
        _mmap_child(args.mmap_child[0], args.mmap_child[1], 10, args.trials)
        return 0

    b = box()
    print("=" * 74)
    print("remax query-path speed — a benchmark, not a gate")
    print("=" * 74)
    for key, val in b.items():
        print(f"  {key}: {val}")
    print(f"  trials: min of {args.trials} (after 1 warmup)")
    print()

    results: dict = {"box": b, "trials": args.trials}

    if "scan" in args.only:
        print("## 1. Threaded scan (identical kernel, identical call)\n")
        rows = bench_scan(args.trials)
        results["scan"] = rows
        print(table(rows, ["n", "B", "threads", "serial_ms", "ms", "speedup",
                           "gb_s", "bit_identical"]))
        print()

    if "select" in args.only:
        print("## 2. Counting select vs argpartition (both exact, same "
              "permutation)\n")
        rows = bench_select(args.trials)
        results["select"] = rows
        print(table(rows, ["n", "k", "argpartition_ms", "counting_ms",
                           "speedup", "argpartition_alloc_bytes",
                           "counting_alloc_bytes"]))
        print()

    if "batch" in args.only:
        print("## 3. Blocked batch scan (same kernel, same popcount count)\n")
        rows = bench_batch(args.trials)
        results["batch"] = rows
        print(table(rows, ["n", "B", "m", "block_rows", "unblocked_t1_ms",
                           "unblocked_t4_ms", "blocked_t1_ms", "blocked_t4_ms",
                           "block_only", "threads_only", "both",
                           "block_on_top_of_threads", "bit_identical"]))
        print()

    if "asym" in args.only:
        print("## 4. Asymmetric table hoist (identical tables)\n")
        rows = bench_asym(args.trials)
        results["asym"] = rows
        build = [r for r in rows if r.get("kind") != "proportion"]
        prop = [r for r in rows if r.get("kind") == "proportion"]
        print(table(build, ["d", "m", "per_query_us", "batched_us", "speedup",
                            "bit_identical"]))
        print()
        print(table(prop, ["d", "n", "table_us", "scan_us", "asym_total_us",
                           "hamming_us", "table_share_pct",
                           "asym_vs_hamming"]))
        print()

    if "mmap" in args.only:
        print(f"## 5. mmap vs load residency (n={args.mmap_n:,} "
              f"d={args.mmap_d})\n")
        rows = bench_mmap(args.trials, args.mmap_n, args.mmap_d)
        results["mmap"] = rows
        print(table(rows, ["residency", "n", "cold_cache", "open_ms",
                           "first_query_ms", "warm_query_ms",
                           "rss_open_delta_mb", "rss_end_mb"]))
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
