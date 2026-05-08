"""Encode-throughput comparison: f32 (default) vs f64 quantizer.

Measures the encode hot path on synthetic data at SPECTER2-typical dim.
The output is bit codes either way; the only difference is the precision
of the rotation matmul. f32 should be roughly 2x faster (sgemm vs dgemm)
on any BLAS, with f32 matrices using half the memory bandwidth.

Usage
-----
    python bench/bench_dtype.py
    python bench/bench_dtype.py --n 50000 --d 768 --k 4 --reps 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from remax import SignBitQuantizer, StackedSignBitQuantizer


def _time(fn, reps: int) -> float:
    """Best-of-reps wall-clock time in seconds."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="remax encode dtype benchmark")
    ap.add_argument("--n", type=int, default=20000, help="corpus size")
    ap.add_argument("--d", type=int, default=768, help="embedding dim (mult of 8)")
    ap.add_argument("--k", type=int, default=4, help="stacked k (0 = 1-bit only)")
    ap.add_argument("--reps", type=int, default=3, help="best-of-N timing reps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.d % 8 != 0:
        raise SystemExit(f"--d must be divisible by 8, got {args.d}")

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.d)).astype(np.float32)

    print(f"[bench_dtype] n={args.n}, d={args.d}, reps={args.reps}")
    print(f"[bench_dtype] input dtype: {X.dtype}")

    # 1-bit
    q32 = SignBitQuantizer(d=args.d, seed=args.seed, dtype=np.float32)
    q64 = SignBitQuantizer(d=args.d, seed=args.seed, dtype=np.float64)
    t32 = _time(lambda: q32.encode(X), args.reps)
    t64 = _time(lambda: q64.encode(X), args.reps)
    speedup = t64 / t32 if t32 > 0 else float("inf")
    print()
    print(f"  SignBitQuantizer (1-bit)")
    print(f"    f32: {t32 * 1000:7.1f} ms  ({args.n / t32 / 1e3:7.1f} k vec/s)")
    print(f"    f64: {t64 * 1000:7.1f} ms  ({args.n / t64 / 1e3:7.1f} k vec/s)")
    print(f"    speedup f32 vs f64: {speedup:.2f}x")

    # Stacked
    if args.k > 0:
        s32 = StackedSignBitQuantizer(d=args.d, k=args.k, seed=args.seed,
                                       dtype=np.float32)
        s64 = StackedSignBitQuantizer(d=args.d, k=args.k, seed=args.seed,
                                       dtype=np.float64)
        t32s = _time(lambda: s32.encode(X), args.reps)
        t64s = _time(lambda: s64.encode(X), args.reps)
        speedup_s = t64s / t32s if t32s > 0 else float("inf")
        print()
        print(f"  StackedSignBitQuantizer (k={args.k})")
        print(f"    f32: {t32s * 1000:7.1f} ms  ({args.n / t32s / 1e3:7.1f} k vec/s)")
        print(f"    f64: {t64s * 1000:7.1f} ms  ({args.n / t64s / 1e3:7.1f} k vec/s)")
        print(f"    speedup f32 vs f64: {speedup_s:.2f}x")

    # Memory footprint of the rotation matrix
    bytes32 = q32.rotation_.nbytes
    bytes64 = q64.rotation_.nbytes
    print()
    print(f"  Rotation matrix size: f32 {bytes32 / 1024:.1f} KiB, "
          f"f64 {bytes64 / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
