#!/usr/bin/env python3
"""Measure the native POPCNT scan against the NumPy LUT fallback.

This is a *benchmark*, not a gate. Wall-clock has no anchor outside the box
it was measured on — there is no published constant for how fast this ought
to run — so the discipline here is benchmarking discipline (matched
implementation effort on both arms, min-of-trials rather than mean, stated
hardware, several corpus sizes) and the output is a table you quote with its
conditions attached, not a threshold anything is allowed to fail.

Why it exists
-------------
``remax/_native.py`` claimed "roughly 50-60x faster than the NumPy LUT path"
while README.md claimed "23x over NumPy (9.7 GB/s)". Both cannot be right as
stated, and neither said at what ``n`` — which is the whole problem, because
the ratio depends strongly on ``n``:

* The NumPy path materialises an ``(n, B)`` uint16 popcount-LUT gather. That
  is 2 bytes of intermediate per input byte, i.e. it touches ~3x the index and
  falls out of cache as soon as ``n * B`` does.
* The native path streams the index once, 8 bytes per ``__builtin_popcountll``,
  and writes 4 bytes per row.

So the small-``n`` ratio is a fixed-overhead-dominated number and the large-``n``
ratio is a bandwidth number, and quoting either one without ``n`` invites the
reader to apply it at the other end of the range.

Usage
-----
    python3 bench/native_speedup.py
    python3 bench/native_speedup.py --d 768 --repeats 9
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from remax import _native
from remax.packing import POPCOUNT_LUT

# Sizes span the range the library actually gets used at: the 10k the
# published benchmarks run, and the 1M the scaling projections talk about.
DEFAULT_SIZES = (10_000, 100_000, 1_000_000)


def numpy_lut(codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The fallback path, verbatim from packing.hamming_distances."""
    xor = np.bitwise_xor(codes, q[None, :])
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)


def _min_of(fn, *args, repeats: int) -> float:
    """Minimum wall-clock over `repeats` runs, after one warm-up.

    Minimum rather than mean: the distribution is one-sided (the machine can
    only steal time, never give it back), so the min is the cleanest estimate
    of the work itself on a shared box.
    """
    fn(*args)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=768, help="embedding dimension")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
        help="corpus row counts",
    )
    args = ap.parse_args(argv)

    if not _native.AVAILABLE:
        print("native scan unavailable on this box; nothing to compare")
        return 1

    B = args.d // 8
    rng = np.random.default_rng(0)

    print(f"machine   : {platform.platform()}")
    print(f"python    : {sys.version.split()[0]}   numpy: {np.__version__}")
    print(f"d={args.d} (B={B} bytes/code)   min of {args.repeats} runs")
    print()
    print(f"{'n':>10}  {'index':>9}  {'native':>10}  {'numpy':>10}  "
          f"{'speedup':>8}  {'GB/s':>7}  {'memcpy':>8}")
    print("-" * 74)

    for n in args.sizes:
        codes = rng.integers(0, 256, size=(n, B), dtype=np.uint8)
        q = rng.integers(0, 256, size=B, dtype=np.uint8)

        # Correctness first: a speedup over a wrong answer is not a speedup.
        assert np.array_equal(
            _native.hamming_distances_native(codes, q), numpy_lut(codes, q)
        ), f"native and numpy disagree at n={n}"

        t_nat = _min_of(_native.hamming_distances_native, codes, q,
                        repeats=args.repeats)
        t_np = _min_of(numpy_lut, codes, q, repeats=args.repeats)

        # memcpy reference, for scale only — see the caveat printed below.
        dst = np.empty_like(codes)
        t_cp = _min_of(np.copyto, dst, codes, repeats=args.repeats)

        gb = codes.nbytes / 1e9
        print(f"{n:>10,}  {gb * 1000:>7.1f} MB  {t_nat * 1e3:>8.2f} ms  "
              f"{t_np * 1e3:>8.2f} ms  {t_np / t_nat:>7.1f}x  "
              f"{gb / t_nat:>6.1f}  {2 * gb / t_cp:>6.1f}")

    print()
    print("speedup = numpy / native. GB/s = index bytes scanned / native time.")
    print("memcpy  = np.copyto over the same buffer, counting read+write, so it")
    print("          is NOT a like-for-like ceiling: the scan reads n*B bytes")
    print("          and writes only 4n, while memcpy moves 2*n*B. Quote it as")
    print("          an order-of-magnitude reference, not as a headroom figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
