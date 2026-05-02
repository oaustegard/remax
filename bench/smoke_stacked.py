"""Smoke benchmark for StackedSignBitQuantizer on synthetic Gaussian data.

Demonstrates the rank-correct precision ladder: doubling ``k`` doubles the
code width and reduces the SimHash estimator's variance by 1/k, climbing
recall@10 monotonically with no broken middle.

Real-embedding benchmarking (SPECTER2 / MiniLM / GloVe) is tracked in
issue #4. This script runs in pure numpy on synthetic data — no model
downloads, no network — and serves as the v0.1.0 checkpoint that stacking
behaves as advertised end-to-end.

Defaults reproduce the issue #3 spec (d=768, n=10k) and are slow enough
that you'll want a few seconds — pass ``--n 2000 --d 256`` for a faster
pass.

Usage
-----
    python bench/smoke_stacked.py
    python bench/smoke_stacked.py --n 2000 --d 256
    python bench/smoke_stacked.py --ks 1 2 4 8 16 --queries 200

Pass criterion: R@10 strictly increases across the ``--ks`` ladder.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from remax import StackedSignBitQuantizer


def recall_at_k(true_topk: np.ndarray, pred_topk: np.ndarray) -> float:
    """Mean ``|true ∩ pred| / k`` over queries."""
    k = true_topk.shape[1]
    total = 0
    for tt, pp in zip(true_topk, pred_topk):
        total += np.intersect1d(tt, pp, assume_unique=False).size
    return total / (true_topk.shape[0] * k)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="remax k-stack smoke benchmark (low-rank Gaussian)."
    )
    ap.add_argument("--n", type=int, default=10000, help="corpus size")
    ap.add_argument("--d", type=int, default=768, help="embedding dim (mult of 8)")
    ap.add_argument("--queries", type=int, default=200, help="number of queries")
    ap.add_argument("--top-k", type=int, default=10, help="top-k for recall")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument(
        "--subdim",
        type=int,
        default=32,
        help=(
            "intrinsic dimensionality of the synthetic data. Lower = wider "
            "cosine spread = easier task. 32 gives R@10 ≈ 0.67 at d=768, k=1."
        ),
    )
    ap.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="stack counts to sweep (default: 1 2 4 8)",
    )
    args = ap.parse_args()

    if args.d % 8 != 0:
        raise SystemExit(f"--d must be divisible by 8, got {args.d}")
    if args.subdim < 1 or args.subdim > args.d:
        raise SystemExit(f"--subdim must be in [1, {args.d}], got {args.subdim}")
    if any(k <= 0 for k in args.ks):
        raise SystemExit(f"--ks values must be positive, got {args.ks}")

    print(
        f"[remax stacked smoke] generating low-rank Gaussian: "
        f"n={args.n}, d={args.d}, subdim={args.subdim}, queries={args.queries}"
    )
    rng = np.random.default_rng(args.seed)

    # Random orthonormal basis for a `subdim`-dim subspace of R^d.
    # Crucially, queries and corpus share this basis — otherwise they live
    # in (nearly) orthogonal subspaces and recall collapses to chance.
    basis, _ = np.linalg.qr(rng.standard_normal((args.d, args.subdim)))
    X = (rng.standard_normal((args.n, args.subdim)) @ basis.T).astype(np.float64)
    Q = (rng.standard_normal((args.queries, args.subdim)) @ basis.T).astype(np.float64)

    # Float ground truth (cosine).
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    sims = Qn @ Xn.T  # (queries, n)
    true_topk = np.argpartition(-sims, args.top_k, axis=1)[:, : args.top_k]

    print()
    print(f"  {'k':>3}  {'R@' + str(args.top_k):>7}  "
          f"{'B/vec':>5}  {'encode':>9}  {'search':>9}")
    print(f"  {'-'*3:>3}  {'-'*7:>7}  {'-'*5:>5}  {'-'*9:>9}  {'-'*9:>9}")

    recalls = []
    for k_stacks in args.ks:
        t0 = time.perf_counter()
        q = StackedSignBitQuantizer(d=args.d, k=k_stacks, seed=args.seed)
        codes = q.encode(X)
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_topk = q.search(Q, codes, k=args.top_k)
        t_search = time.perf_counter() - t0

        r = recall_at_k(true_topk, pred_topk)
        recalls.append(r)
        bytes_per_vec = codes.shape[1]
        print(
            f"  {k_stacks:>3}  {r:>7.4f}  {bytes_per_vec:>5d}  "
            f"{t_encode * 1000:>7.1f}ms  {t_search * 1000:>7.1f}ms"
        )

    print()

    # Pass criterion: strict monotonicity in k.
    if not all(b > a for a, b in zip(recalls, recalls[1:])):
        raise SystemExit(
            f"R@{args.top_k} not strictly monotone in k: "
            f"{dict(zip(args.ks, [round(r, 4) for r in recalls]))}"
        )

    print(f"[remax stacked smoke] OK — R@{args.top_k} monotone across "
          f"k={tuple(args.ks)}")


if __name__ == "__main__":
    main()
