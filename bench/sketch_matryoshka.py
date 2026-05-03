"""Post-hoc Matryoshka: can sketching back into dimensional reduction?

Experiment comparing strategies for producing k-bit binary signatures
from 768-d SPECTER2 embeddings. The question: if you can't retrain your
encoder with Matryoshka loss, can you get useful shorter signatures via
projection + sign-bit extraction?

Strategies tested at each bit budget k ∈ {64, 128, 192, 256, 384, 512, 768}:

  prefix       First k dims of centered embedding, sign-packed.
               Baseline: no projection, no fitting.

  gaussian     Random Gaussian projection R ∈ R^{768×k} scaled 1/√k.
               JL guarantee: pairwise distances preserved within (1±ε).

  countsketch  Sparse random projection: hash each of 768 dims into one
               of k buckets with random sign flip. Same JL class, O(d)
               instead of O(dk) to project.

  pca          Project onto top-k principal components. Fitted to corpus.
               Should be optimal among linear projections — but isn't,
               because sign bits weight all dimensions equally while PCA
               concentrates variance in the top components.

  haar-trunc   Haar random rotation of centered data, truncated to k dims.
               Distributes variance uniformly, making every bit equally
               informative. Best performer at k ≥ 256.

Ground truth: top-10 by float32 inner product on full 768-d embeddings.
Metrics: R@10 and R@100 against ground truth.

Key findings:
  1. Post-hoc dimensional reduction works. At 256 bits (32 bytes/vec,
     96× compression), haar-trunc achieves R@100 = 0.928.
  2. Random projection (gaussian, countsketch) is competitive with prefix
     truncation — no fitting required.
  3. PCA is the worst basis for sign-bit extraction: it concentrates
     variance, making high-k Hamming distances noise-dominated.
  4. PCA whitening doesn't help (sign is invariant to positive scaling)
     and actually hurts by amplifying noise dimensions.
  5. Signal uniformity matters more than variance uniformity. Modern
     encoders distribute signal roughly uniformly across dimensions,
     which is why naive sign extraction works.

Usage:
    # Fetch data first:
    bash bench/fetch_specter2_cache.sh
    # Run:
    python bench/sketch_matryoshka.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from pathlib import Path


def load_specter2(cache_dir: Path | None = None) -> np.ndarray:
    """Load the 10k SPECTER2 embedding cache."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".cache" / "SPECTER2"
    p = cache_dir / "embeddings.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"SPECTER2 cache not found at {p}.\n"
            "Run: bash bench/fetch_specter2_cache.sh"
        )
    return np.load(str(p))


# ── Bit-packing and Hamming search ────────────────────────────────────

POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def sign_pack(X: np.ndarray) -> np.ndarray:
    """sign(X) → packed uint8, with zero-padding to multiple of 8 dims."""
    if X.ndim == 1:
        X = X[None, :]
    n, d = X.shape
    pad = (8 - d % 8) % 8
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)))
    return np.packbits(X > 0, axis=1)


def hamming_recall_at(
    q_codes: np.ndarray,
    c_codes: np.ndarray,
    truth_k: np.ndarray,
    Ns: list[int],
) -> dict[int, float]:
    """Compute R@N for multiple N values from one distance computation."""
    nq = q_codes.shape[0]
    k = truth_k.shape[1]
    max_N = max(Ns)
    preds = {N: [] for N in Ns}

    for i in range(nq):
        d = POPCOUNT_LUT[np.bitwise_xor(c_codes, q_codes[i])].sum(1)
        idx = np.argpartition(d, max_N)[:max_N]
        order = idx[np.argsort(d[idx])]
        for N in Ns:
            preds[N].append(set(order[:N].tolist()))

    results = {}
    for N in Ns:
        hits = sum(
            len(p & set(truth_k[i].tolist()))
            for i, p in enumerate(preds[N])
        )
        results[N] = hits / (nq * k)
    return results


# ── Projection strategies ─────────────────────────────────────────────

def project_prefix(X_centered: np.ndarray, k: int, **_) -> np.ndarray:
    return X_centered[:, :k]


def project_gaussian(X_centered: np.ndarray, k: int, *, seed: int = 42, **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((X_centered.shape[1], k)).astype(np.float32) / np.sqrt(k)
    return X_centered @ R


def project_countsketch(X_centered: np.ndarray, k: int, *, seed: int = 43, **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = X_centered.shape[1]
    buckets = rng.integers(0, k, size=d)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)
    S = np.zeros((d, k), dtype=np.float32)
    S[np.arange(d), buckets] = signs
    return X_centered @ S


def project_pca(X_centered: np.ndarray, k: int, *, Vt: np.ndarray, **_) -> np.ndarray:
    return X_centered @ Vt[:k].T


def project_haar_trunc(X_centered: np.ndarray, k: int, *, H: np.ndarray, **_) -> np.ndarray:
    return (X_centered @ H)[:, :k]


STRATEGIES = {
    "prefix": project_prefix,
    "gaussian": project_gaussian,
    "countsketch": project_countsketch,
    "pca": project_pca,
    "haar-trunc": project_haar_trunc,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[64, 128, 192, 256, 384, 512, 768],
        help="Bit budgets to test (default: 64 128 192 256 384 512 768)",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES.keys()),
        choices=list(STRATEGIES.keys()),
        help="Strategies to test",
    )
    ap.add_argument("--seed", type=int, default=99, help="Split RNG seed")
    ap.add_argument("--queries", type=int, default=100, help="Number of held-out queries")
    args = ap.parse_args()

    X = load_specter2()
    d = X.shape[1]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    queries, corpus = X[perm[: args.queries]], X[perm[args.queries :]]

    # Ground truth: top-10 by inner product
    truth10 = np.argsort(-(queries @ corpus.T), axis=1)[:, :10]

    # Centering
    mu = corpus.mean(0)
    corpus_c = (corpus - mu).astype(np.float32)
    queries_c = (queries - mu).astype(np.float32)

    # Precompute fitted projections
    try:
        from remax.rotation import haar_rotation
        H = haar_rotation(d, seed=args.seed)
    except ImportError:
        H = None
        if "haar-trunc" in args.strategies:
            print("warning: remax not installed, skipping haar-trunc", file=sys.stderr)
            args.strategies = [s for s in args.strategies if s != "haar-trunc"]

    _, S, Vt = np.linalg.svd(corpus_c, full_matrices=False)

    Ns = [10, 100]

    # Header
    print(f"{'strategy':<16} {'k':>4} {'bits':>5} {'R@10':>7} {'R@100':>7}  {'B/vec':>5}  {'ratio':>6}")
    print("─" * 62)

    t0 = time.perf_counter()
    for ki, k in enumerate(args.ks):
        for name in args.strategies:
            fn = STRATEGIES[name]
            kwargs = {"seed": args.seed, "Vt": Vt, "H": H}
            cp = fn(corpus_c, k, **kwargs)
            qp = fn(queries_c, k, **kwargs)

            c_codes = sign_pack(cp)
            q_codes = sign_pack(qp)
            recalls = hamming_recall_at(q_codes, c_codes, truth10, Ns)

            bpv = c_codes.shape[1]
            ratio = f"{d * 4 / bpv:.0f}x"
            print(
                f"{name:<16} {k:>4} {k:>5} "
                f"{recalls[10]:>7.3f} {recalls[100]:>7.3f}  "
                f"{bpv:>5}  {ratio:>6}"
            )

        if ki < len(args.ks) - 1:
            print()

    # Full 768-bit centered baseline (from blog post)
    print()
    print("─" * 62)
    c_full = np.packbits(corpus_c > 0, axis=1)
    q_full = np.packbits(queries_c > 0, axis=1)
    r_full = hamming_recall_at(q_full, c_full, truth10, Ns)
    print(
        f"{'blog-baseline':<16} {768:>4} {768:>5} "
        f"{r_full[10]:>7.3f} {r_full[100]:>7.3f}  "
        f"{c_full.shape[1]:>5}  {'32x':>6}"
    )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
