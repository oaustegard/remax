"""Smoke benchmark for SignBitQuantizer on synthetic Gaussian data.

Runs without external corpora — exists to verify the v0.1.0 1-bit core
is working end-to-end on a distribution where the recall-vs-cosine
relation is well-defined. Real-embedding benchmarking (SPECTER2 etc.)
is tracked in issue #4.

Why low-rank Gaussian, not isotropic
------------------------------------
A pure isotropic Gaussian at d=768 concentrates pairwise cosines around
zero with stdev ≈ 1/√d ≈ 0.036, which is the same order as SimHash's
own noise floor at 768 bits. Recall@10 on such data caps near chance
regardless of encoder quality, so it tells us nothing.

A Gaussian *mixture* with unit-norm centers fixes one half (cluster
identification works) but creates the opposite problem: within a tight
cluster, all 50 cluster-mates have cosines packed in a ≈0.015-wide band,
which is below the rank resolution of any 768-bit code. Recall@10
plateaus around 0.3 because the encoder can't tie-break correctly.

A low-rank Gaussian — data drawn from a ``subdim``-dimensional Gaussian
embedded in ``d`` via a random orthonormal basis — gives a continuous,
non-degenerate spread of cosines. Real text embeddings concentrate on
a low-dimensional manifold for the same reason, so this is also closer
to the operating regime that motivates the library.

Usage
-----
    python bench/smoke_1bit.py
    python bench/smoke_1bit.py --n 5000 --d 768 --queries 200 --seed 42

Pass criterion: ``R@10 > 0.5`` at the default config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from remax import SignBitQuantizer


def recall_at_k(true_topk: np.ndarray, pred_topk: np.ndarray) -> float:
    """Mean ``|true ∩ pred| / k`` over queries."""
    k = true_topk.shape[1]
    total = 0
    for tt, pp in zip(true_topk, pred_topk):
        total += np.intersect1d(tt, pp, assume_unique=False).size
    return total / (true_topk.shape[0] * k)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="remax 1-bit smoke benchmark (low-rank Gaussian)."
    )
    ap.add_argument("--n", type=int, default=1000, help="corpus size")
    ap.add_argument("--d", type=int, default=768, help="embedding dim (mult of 8)")
    ap.add_argument("--queries", type=int, default=100, help="number of queries")
    ap.add_argument("--k", type=int, default=10, help="top-k")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument(
        "--subdim",
        type=int,
        default=32,
        help=(
            "intrinsic dimensionality of the synthetic data. Lower = wider "
            "cosine spread = easier task. 32 gives R@10 ≈ 0.7 at d=768."
        ),
    )
    args = ap.parse_args()

    if args.d % 8 != 0:
        raise SystemExit(f"--d must be divisible by 8, got {args.d}")
    if args.subdim < 1 or args.subdim > args.d:
        raise SystemExit(f"--subdim must be in [1, {args.d}], got {args.subdim}")

    print(
        f"[remax smoke] generating low-rank Gaussian: "
        f"n={args.n}, d={args.d}, subdim={args.subdim}"
    )
    rng = np.random.default_rng(args.seed)

    # Random orthonormal basis for a `subdim`-dim subspace of R^d.
    basis, _ = np.linalg.qr(rng.standard_normal((args.d, args.subdim)))  # (d, subdim)

    # Draw points iid in the subspace, then embed.
    X = (rng.standard_normal((args.n, args.subdim)) @ basis.T).astype(np.float64)
    Q = (rng.standard_normal((args.queries, args.subdim)) @ basis.T).astype(np.float64)

    # Float ground truth (cosine).
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    sims = Qn @ Xn.T  # (queries, n)
    true_topk = np.argpartition(-sims, args.k, axis=1)[:, : args.k]

    # SimHash 1-bit.
    t0 = time.perf_counter()
    q = SignBitQuantizer(d=args.d, seed=args.seed)
    codes = q.encode(X)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_topk = q.search(Q, codes, k=args.k)
    t_search = time.perf_counter() - t0

    r = recall_at_k(true_topk, pred_topk)
    bytes_per_vec = codes.shape[1]
    print(
        f"[remax smoke] encoded {args.n}×{args.d} → {codes.shape} "
        f"({bytes_per_vec} B/vec) in {t_encode * 1000:.1f} ms"
    )
    print(
        f"[remax smoke] searched {args.queries} queries in "
        f"{t_search * 1000:.1f} ms ({1000 * t_search / args.queries:.2f} ms/q)"
    )
    print(f"[remax smoke] R@{args.k} = {r:.3f}")

    if r <= 0.5:
        raise SystemExit(
            f"R@{args.k}={r:.3f} below smoke threshold of 0.5 — something is wrong"
        )
    print("[remax smoke] OK")


if __name__ == "__main__":
    main()
