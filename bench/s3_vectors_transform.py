"""Which transform should you push to a cosine-metric managed ANN index?

Motivation
----------
S3 Vectors (and every managed ANN service shaped like it) stores ``float32``
and scores with ``cosine`` or ``euclidean``. It does not offer raw inner
product. remax's binary codes have no place in such an index at all — so on
that path remax's contribution is reduced to the *transform*: center and/or
truncate before upload, and apply the identical transform to the query.

``bench/results/SKETCH_MATRYOSHKA.md`` already measured center+truncate, but
it measured it under an **inner-product** scan (``f32-centered``, k=256,
R@100 = 0.943). That number cannot be carried onto a cosine index: cosine
normalizes away the vector norms that IP is scoring with, so it is a
different ranking function over the same stored bytes. This driver measures
the transform under the metric the index actually computes.

The centering question is the one that matters, because centering is the
single biggest lever on the *binary* path (+0.324 R@100 at k=64,
SKETCH_MATRYOSHKA.md finding 2) and it is tempting to assume it carries over.

What is measured
----------------
A 2x2x2: {truncate-only, center+truncate} x {ip, cosine} x ground truth
{full-768 ip, full-768 cosine}, swept over dimension and repeated across
seeds. Each cell is a full retrieval run, not a proxy.

``cosine`` rows L2-normalize both corpus and query before an inner-product
scan, which is exactly the ranking a ``distanceMetric="cosine"`` index
computes — cosine is scale-invariant, so normalizing up front changes the
scores but not the order.

Recall convention follows the rest of this bench: ground truth is the top-10
by full-dimension float32 similarity, and ``R@N`` is the fraction of that
top-10 appearing in the returned top-N. ``rerank R@10`` re-scores the
returned top-100 by exact full-768 float32 IP and keeps 10 — the stage-2
pattern from ``docs/specter2-search-pipeline.md``.

Usage
-----
    bash bench/fetch_specter2_cache.sh
    python bench/s3_vectors_transform.py
    python bench/s3_vectors_transform.py --csv bench/results/s3_vectors_transform.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.datasets import load_dataset  # noqa: E402

DIMS = (128, 192, 256, 384, 512, 768)
SEEDS = (99, 1, 2, 3, 4, 5, 6, 7)
N_QUERIES = 100
TRUTH_K = 10


def unit(A: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero rows are left at zero rather than nan."""
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def top_n(queries: np.ndarray, corpus: np.ndarray, n: int) -> np.ndarray:
    """Top-n by inner product, ordered best-first."""
    sims = queries @ corpus.T
    n = min(n, corpus.shape[0])
    idx = np.argpartition(-sims, n - 1, axis=1)[:, :n]
    out = np.empty((queries.shape[0], n), dtype=np.intp)
    for i in range(queries.shape[0]):
        out[i] = idx[i][np.argsort(-sims[i, idx[i]])]
    return out


def recall(truth: np.ndarray, pred: np.ndarray, n: int) -> float:
    hits = sum(
        len(set(truth[i].tolist()) & set(pred[i, :n].tolist()))
        for i in range(truth.shape[0])
    )
    return hits / (truth.shape[0] * truth.shape[1])


def run_seed(X: np.ndarray, seed: int, dims) -> list[dict]:
    """One train/query split; returns a row per (dim, transform, metric, gt)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    queries, corpus = X[perm[:N_QUERIES]], X[perm[N_QUERIES:]]

    truths = {
        "ip": np.argsort(-(queries @ corpus.T), axis=1)[:, :TRUTH_K],
        "cosine": np.argsort(
            -(unit(queries) @ unit(corpus).T), axis=1
        )[:, :TRUTH_K],
    }

    mu = corpus.mean(0)
    corpus_c = (corpus - mu).astype(np.float32)
    queries_c = (queries - mu).astype(np.float32)

    rows = []
    for dim in dims:
        transforms = {
            "truncate": (corpus[:, :dim], queries[:, :dim]),
            "center+truncate": (corpus_c[:, :dim], queries_c[:, :dim]),
        }
        for tname, (C, Q) in transforms.items():
            for metric in ("ip", "cosine"):
                Ci, Qi = (unit(C), unit(Q)) if metric == "cosine" else (C, Q)
                pred = top_n(Qi, Ci, 100)

                # Stage 2: exact full-768 float32 IP over the 100 candidates.
                hits = 0
                for i in range(N_QUERIES):
                    cand = pred[i]
                    order = cand[np.argsort(-(corpus[cand] @ queries[i]))][:TRUTH_K]
                    hits += len(set(order.tolist()) & set(truths["ip"][i].tolist()))
                rerank_r10 = hits / (N_QUERIES * TRUTH_K)

                for gt, truth in truths.items():
                    rows.append({
                        "seed": seed,
                        "dim": dim,
                        "transform": tname,
                        "metric": metric,
                        "ground_truth": gt,
                        "r10": recall(truth, pred, 10),
                        "r100": recall(truth, pred, 100),
                        # rerank is scored against IP ground truth only: it
                        # re-scores with full-768 IP, so that is the target
                        # it is by construction chasing.
                        "rerank_r10": rerank_r10 if gt == "ip" else float("nan"),
                    })
    return rows


def aggregate(rows: list[dict], **where) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return (
        np.array([r["r10"] for r in sel]),
        np.array([r["r100"] for r in sel]),
        np.array([r["rerank_r10"] for r in sel]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="SPECTER2")
    ap.add_argument("--dims", nargs="+", type=int, default=list(DIMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--csv", type=str, default=None, help="write per-seed rows here")
    args = ap.parse_args()

    X, info = load_dataset(args.dataset)
    print(f"# dataset={info['name']} shape=({info['n']}, {info['dim']}) "
          f"queries={N_QUERIES} seeds={len(args.seeds)}")
    nrm = np.linalg.norm(X, axis=1)
    print(f"# raw norms: mean={nrm.mean():.3f} std={nrm.std():.3f} "
          f"(cv={nrm.std() / nrm.mean():.4f})")
    print()

    rows = []
    for seed in args.seeds:
        rows += run_seed(X, seed, args.dims)

    # ── Main sweep, against raw-IP ground truth (this bench's convention) ──
    print("Ground truth: top-10 by full-768 float32 inner product")
    print(f"{'transform':<17} {'metric':>7} {'dim':>5} {'R@10':>14} "
          f"{'R@100':>14} {'rerank R@10':>14}")
    print("-" * 76)
    for dim in args.dims:
        for tname in ("truncate", "center+truncate"):
            for metric in ("ip", "cosine"):
                r10, r100, rr = aggregate(
                    rows, dim=dim, transform=tname, metric=metric,
                    ground_truth="ip",
                )
                print(f"{tname:<17} {metric:>7} {dim:>5} "
                      f"{r10.mean():>7.3f} ±{r10.std():.3f} "
                      f"{r100.mean():>7.3f} ±{r100.std():.3f} "
                      f"{rr.mean():>7.3f} ±{rr.std():.3f}")
        print()

    # ── The centering question, both ground truths, at the operating point ──
    print("-" * 76)
    print("Centering, head to head (positive = truncate-only wins)")
    print()
    for gt in ("ip", "cosine"):
        print(f"  ground truth: full-768 {gt}")
        for dim in args.dims:
            a10, a100, _ = aggregate(
                rows, dim=dim, transform="truncate",
                metric="cosine", ground_truth=gt,
            )
            b10, b100, _ = aggregate(
                rows, dim=dim, transform="center+truncate",
                metric="cosine", ground_truth=gt,
            )
            d10, d100 = a10 - b10, a100 - b100
            print(f"    dim={dim:<4} ΔR@10={d10.mean():>+7.3f}  "
                  f"ΔR@100={d100.mean():>+7.3f}  "
                  f"seeds won: {int((d10 > 0).sum())}/{len(d10)} (R@10), "
                  f"{int((d100 > 0).sum())}/{len(d100)} (R@100)")
        print()

    if args.csv:
        dest = Path(args.csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {dest}")


if __name__ == "__main__":
    main()
