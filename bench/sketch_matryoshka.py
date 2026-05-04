"""Post-hoc Matryoshka: can sketching back into dimensional reduction?

Compares strategies for producing k-dimensional signatures from a
fixed-d encoder. The question: without retraining, can you get a
useful shorter representation via projection + sign-bit extraction?

Two layers of comparison at each bit budget k:

Float32 baselines (what truncation alone costs):
  f32-raw        Truncate raw embedding to k dims, float32 IP search.
  f32-centered   Center then truncate to k dims, float32 IP search.

1-bit strategies (the compression question):
  sign-raw       sign(x[:k]). No centering, no rotation. Truly free.
  sign-centered  sign((x-mu)[:k]). Center-only. Still free.
  gaussian       Random Gaussian projection to k dims, sign-packed.
  countsketch    Sparse random projection to k dims, sign-packed.
  pca            PCA projection to k dims, sign-packed.
  haar-trunc     Haar rotate centered data, truncate to k, sign-packed.

Plus full-width baselines including remex at precision=1 and precision=4
(remex uses its own search function with codebook and stored norms, so
per-k truncation comparisons wouldn't be apples-to-apples).

Datasets:
  SPECTER2  — 768-d, not Matryoshka-trained, norms ~20-22.
  GEMINI    — 3072-d, Matryoshka-trained (down to 768-d), L2-normalized.

Index-selection grid (--index-grid): at specific ks, compare prefix /
suffix / spaced / random index selection on centered sign-packed data.
The interesting question for Matryoshka encoders: does training make
the prefix specifically informative, or is any subset equivalent?

Ground truth: top-10 by float32 inner product on full d.

Usage:
    bash bench/fetch_specter2_cache.sh   # or fetch_gemini_cache.sh
    python bench/sketch_matryoshka.py --dataset SPECTER2
    python bench/sketch_matryoshka.py --dataset GEMINI
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from pathlib import Path


DATASET_DIMS = {"SPECTER2": 768, "GEMINI": 3072}

DATASET_KS = {
    "SPECTER2": [64, 128, 192, 256, 384, 512, 768],
    "GEMINI": [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072],
}

INDEX_GRID_KS = {
    "SPECTER2": [256, 768],
    "GEMINI": [256, 768],
}


def load_dataset(name: str, cache_dir: Path | None = None) -> np.ndarray:
    if name not in DATASET_DIMS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASET_DIMS)}")
    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".cache" / name
    p = cache_dir / "embeddings.npy"
    if not p.exists():
        script = "fetch_gemini_cache.sh" if name == "GEMINI" else "fetch_specter2_cache.sh"
        raise FileNotFoundError(
            f"{name} cache not found at {p}.\n"
            f"Run: bash bench/{script}"
        )
    arr = np.load(str(p))
    expected_d = DATASET_DIMS[name]
    if arr.ndim != 2 or arr.shape[1] != expected_d:
        raise ValueError(
            f"{name} cache has shape {arr.shape}; expected (*, {expected_d})"
        )
    return arr


# ── Search primitives ─────────────────────────────────────────────────

POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def sign_pack(X: np.ndarray) -> np.ndarray:
    """sign(X) → packed uint8, zero-padding to multiple of 8 dims."""
    if X.ndim == 1:
        X = X[None, :]
    pad = (8 - X.shape[1] % 8) % 8
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)))
    return np.packbits(X > 0, axis=1)


def recall_at(truth_k: np.ndarray, pred: np.ndarray) -> float:
    k = truth_k.shape[1]
    hits = sum(
        len(set(truth_k[i].tolist()) & set(pred[i].tolist()))
        for i in range(truth_k.shape[0])
    )
    return hits / (truth_k.shape[0] * k)


def float32_topN(queries: np.ndarray, corpus: np.ndarray, N: int) -> np.ndarray:
    sims = queries @ corpus.T
    nq = queries.shape[0]
    N = min(N, corpus.shape[0])
    idx = np.argpartition(-sims, N, axis=1)[:, :N]
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        out[i] = idx[i][np.argsort(-sims[i, idx[i]])]
    return out


def hamming_topN(q_codes: np.ndarray, c_codes: np.ndarray, N: int) -> np.ndarray:
    nq = q_codes.shape[0]
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        d = POPCOUNT_LUT[np.bitwise_xor(c_codes, q_codes[i])].sum(1)
        idx = np.argpartition(d, N)[:N]
        out[i] = idx[np.argsort(d[idx])]
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dataset", choices=list(DATASET_DIMS), default="SPECTER2",
        help="Embedding source. SPECTER2=768-d, GEMINI=3072-d Matryoshka.",
    )
    ap.add_argument(
        "--ks", nargs="+", type=int, default=None,
        help="Bit budgets to test (default: dataset-specific grid).",
    )
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument(
        "--index-grid", action="store_true",
        help="Also run prefix/suffix/spaced/random index-selection comparison.",
    )
    ap.add_argument(
        "--n-random", type=int, default=5,
        help="Number of random-index trials in the index-selection grid.",
    )
    args = ap.parse_args()

    ks = args.ks if args.ks is not None else DATASET_KS[args.dataset]

    X = load_dataset(args.dataset)
    d = X.shape[1]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    queries, corpus = X[perm[:args.queries]], X[perm[args.queries:]]

    print(f"# Dataset: {args.dataset}  shape={X.shape}  dtype={X.dtype}")
    norms = np.linalg.norm(corpus[:200], axis=1)
    print(f"# Corpus norms (n=200): mean={norms.mean():.3f} std={norms.std():.3f}")
    print()

    # Ground truth: top-10 by full float32 inner product
    truth10 = np.argsort(-(queries @ corpus.T), axis=1)[:, :10]

    mu = corpus.mean(0)
    corpus_c = (corpus - mu).astype(np.float32)
    queries_c = (queries - mu).astype(np.float32)

    # Precompute PCA
    print("# Precomputing PCA (SVD)...", file=sys.stderr)
    t_pca = time.perf_counter()
    _, S, Vt = np.linalg.svd(corpus_c, full_matrices=False)
    print(f"#   SVD took {time.perf_counter() - t_pca:.1f}s", file=sys.stderr)

    # Precompute Haar rotation
    H = None
    try:
        from remax.rotation import haar_rotation
        print(f"# Precomputing Haar rotation at d={d}...", file=sys.stderr)
        t_haar = time.perf_counter()
        H = haar_rotation(d, seed=args.seed)
        print(f"#   Haar took {time.perf_counter() - t_haar:.1f}s", file=sys.stderr)
    except ImportError:
        print("# (remax.rotation not importable, skipping haar-trunc)", file=sys.stderr)

    Ns = [10, 100]  # R@10 and R@100

    def build_strategies(k):
        """Build (name, type, corpus_data, query_data) for this k."""
        strats = []

        # Float32 baselines
        strats.append(("f32-raw", "f32", corpus[:, :k], queries[:, :k]))
        strats.append(("f32-centered", "f32", corpus_c[:, :k], queries_c[:, :k]))

        # 1-bit: truly free (no centering)
        strats.append(("sign-raw", "1bit",
                        sign_pack(corpus[:, :k]), sign_pack(queries[:, :k])))

        # 1-bit: center-only free
        strats.append(("sign-centered", "1bit",
                        sign_pack(corpus_c[:, :k]), sign_pack(queries_c[:, :k])))

        # 1-bit: random Gaussian projection
        gr = np.random.default_rng(args.seed + 1)
        R = gr.standard_normal((d, k)).astype(np.float32) / np.sqrt(k)
        strats.append(("gaussian", "1bit",
                        sign_pack(corpus_c @ R), sign_pack(queries_c @ R)))

        # 1-bit: count-sketch
        cr = np.random.default_rng(args.seed + 2)
        buckets = cr.integers(0, k, size=d)
        signs = cr.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)
        Sk = np.zeros((d, k), dtype=np.float32)
        Sk[np.arange(d), buckets] = signs
        strats.append(("countsketch", "1bit",
                        sign_pack(corpus_c @ Sk), sign_pack(queries_c @ Sk)))

        # 1-bit: PCA (known bad — included for completeness)
        strats.append(("pca", "1bit",
                        sign_pack(corpus_c @ Vt[:k].T),
                        sign_pack(queries_c @ Vt[:k].T)))

        # 1-bit: Haar rotation + truncation
        if H is not None:
            strats.append(("haar-trunc", "1bit",
                            sign_pack((corpus_c @ H)[:, :k]),
                            sign_pack((queries_c @ H)[:, :k])))

        return strats

    # ── Per-k results ─────────────────────────────────────────────
    print(f"{'strategy':<16} {'type':>4} {'k':>4} "
          f"{'R@10':>7} {'R@100':>7}  {'B/vec':>5}  {'ratio':>6}")
    print("─" * 64)

    t0 = time.perf_counter()
    for ki, k in enumerate(ks):
        for name, stype, cdata, qdata in build_strategies(k):
            if stype == "f32":
                bpv = k * 4
                pred = float32_topN(qdata, cdata, max(Ns))
            else:
                bpv = cdata.shape[1]
                pred = hamming_topN(qdata, cdata, max(Ns))

            r10 = recall_at(truth10, pred[:, :10])
            r100 = recall_at(truth10, pred)
            ratio = f"{d * 4 / bpv:.0f}x"
            print(f"{name:<16} {stype:>4} {k:>4} "
                  f"{r10:>7.3f} {r100:>7.3f}  {bpv:>5}  {ratio:>6}")

        if ki < len(ks) - 1:
            print()

    # ── Full-width baselines ──────────────────────────────────────
    print()
    print("─" * 64)
    print(f"Full-width baselines ({d} dims):")
    print()

    # f32-full: ground truth (should be 1.000 by definition)
    pred_full = float32_topN(queries, corpus, 100)
    print(f"{'f32-full':<16} {'f32':>4} {d:>4} "
          f"{recall_at(truth10, pred_full[:, :10]):>7.3f} "
          f"{recall_at(truth10, pred_full):>7.3f}  {d*4:>5}  {'1x':>6}")

    # blog baseline: sign(x - mu), d bits
    c_blog = np.packbits(corpus_c > 0, axis=1)
    q_blog = np.packbits(queries_c > 0, axis=1)
    pred_blog = hamming_topN(q_blog, c_blog, 100)
    print(f"{'blog-baseline':<16} {'1bit':>4} {d:>4} "
          f"{recall_at(truth10, pred_blog[:, :10]):>7.3f} "
          f"{recall_at(truth10, pred_blog):>7.3f}  "
          f"{c_blog.shape[1]:>5}  {d*4 // c_blog.shape[1]:>4}x")

    # remex baselines (use remex's own search with codebook + norms)
    try:
        import remex

        for bits, precs in [(1, [1]), (4, [1, 4])]:
            q_rmx = remex.Quantizer(d=d, bits=bits, seed=args.seed)
            c_rmx = q_rmx.encode(corpus)
            for prec in precs:
                hits10, hits100 = 0, 0
                for i in range(args.queries):
                    idx, _ = q_rmx.search(c_rmx, queries[i], k=100, precision=prec)
                    hits10 += len(set(idx[:10].tolist()) & set(truth10[i].tolist()))
                    hits100 += len(set(idx.tolist()) & set(truth10[i].tolist()))
                r10 = hits10 / (args.queries * 10)
                r100 = hits100 / (args.queries * 10)
                bpv = d * bits // 8 + 4  # packed codes + 4-byte norm
                label = f"remex-{bits}b@p={prec}"
                ratio = f"{d * 4 / bpv:.0f}x"
                print(f"{label:<16} {'rmx':>4} {d:>4} "
                      f"{r10:>7.3f} {r100:>7.3f}  {bpv:>5}  {ratio:>6}")
    except ImportError:
        print("(remex not installed, skipping remex baselines)")

    # ── Index-selection grid ──────────────────────────────────────
    if args.index_grid:
        print()
        print("─" * 64)
        print(f"Index-selection grid (centered, sign-packed; n_random={args.n_random}):")
        print()
        print(f"{'select':<16} {'k':>4} {'R@10':>7} {'R@100':>7}  {'B/vec':>5}")
        print("─" * 48)

        irng = np.random.default_rng(args.seed + 100)
        for k in INDEX_GRID_KS[args.dataset]:
            if k > d:
                continue

            def run_idx(label, idx):
                cc = sign_pack(corpus_c[:, idx])
                qc = sign_pack(queries_c[:, idx])
                pred = hamming_topN(qc, cc, max(Ns))
                r10 = recall_at(truth10, pred[:, :10])
                r100 = recall_at(truth10, pred)
                print(f"{label:<16} {k:>4} {r10:>7.3f} {r100:>7.3f}  {cc.shape[1]:>5}")

            run_idx("prefix", np.arange(k))
            run_idx("suffix", np.arange(d - k, d))
            run_idx("spaced", np.round(np.linspace(0, d - 1, k)).astype(int))

            # Random: average over n_random trials
            r10s, r100s = [], []
            for _ in range(args.n_random):
                idx = irng.choice(d, size=k, replace=False)
                cc = sign_pack(corpus_c[:, idx])
                qc = sign_pack(queries_c[:, idx])
                pred = hamming_topN(qc, cc, max(Ns))
                r10s.append(recall_at(truth10, pred[:, :10]))
                r100s.append(recall_at(truth10, pred))
            bpv_rand = sign_pack(corpus_c[:, :k]).shape[1]
            print(f"{'random (avg)':<16} {k:>4} "
                  f"{np.mean(r10s):>7.3f} {np.mean(r100s):>7.3f}  {bpv_rand:>5}"
                  f"   ±{np.std(r100s):.3f}")
            if k != INDEX_GRID_KS[args.dataset][-1]:
                print()

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
