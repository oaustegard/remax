"""CLI driver for the v0.1.0 baseline.

Produces ``bench/results/BASELINE.md`` from cached real embeddings:

* SPECTER2 (768-d, blog-post reproduction)
* MiniLM-L6-v2 (384-d)
* GloVe-300d (300-d)

Each row reports R@10 vs float32 ground truth at the four v0.1.0 quantizer
configurations: 1-bit (``SignBitQuantizer``) and stacked k=2,4,8
(``StackedSignBitQuantizer``).

Usage
-----
::

    # one-time: fetch the SPECTER2 cache
    bash bench/fetch_specter2_cache.sh

    # run the baseline
    python -m remax.bench.run_baseline
    # or
    python bench/run_baseline.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from remax import SignBitQuantizer, StackedSignBitQuantizer
from remax.bench import datasets
from remax.bench.eval import evaluate_quantizer, exact_knn, recall_at_k
from remax.corpus import Corpus

__all__ = [
    "LADDER_KS",
    "compute_baseline_for_embeddings",
    "format_baseline_md",
    "main",
]

# v0.1.0 ladder. Pinned by issue #4: 1-bit + stacked k=2,4,8.
LADDER_KS: tuple[int, ...] = (2, 4, 8)
DEFAULT_N = 10_000
DEFAULT_N_QUERIES = 100
DEFAULT_K_EVAL = 10
DEFAULT_SEED = 42
# Query-split RNG seed mirrors remex/bench/onebit_experiment.py so the
# blog-post number lands at the same place.
QUERY_SPLIT_SEED = 99


# --------------------------------------------------------------------- #
# Per-embeddings orchestration
# --------------------------------------------------------------------- #


def _split_queries(
    emb: np.ndarray, n_queries: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``emb`` into (corpus, queries) using a permutation seeded with
    ``split_seed`` — same protocol as remex's onebit_experiment to keep
    blog-post numbers reproducible."""
    if n_queries <= 0 or n_queries >= emb.shape[0]:
        raise ValueError(
            f"n_queries={n_queries} out of range for n={emb.shape[0]}"
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(emb.shape[0])
    queries = emb[perm[:n_queries]]
    corpus = emb[perm[n_queries:]]
    return corpus, queries


def compute_baseline_for_embeddings(
    *,
    name: str,
    emb: np.ndarray,
    n_queries: int = DEFAULT_N_QUERIES,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    center: bool = True,
) -> dict:
    """Run the v0.1.0 ladder over a single embeddings array.

    Parameters
    ----------
    name : str
        Dataset label for the output row.
    emb : (n, d) np.ndarray
        Real-valued embeddings. ``d`` must be divisible by 8 (remax codes
        are bit-packed bytewise).
    n_queries : int
        Held-out query count, sampled via ``QUERY_SPLIT_SEED`` permutation.
    k_eval : int
        Top-k cutoff for both search and recall. Issue #4 fixes this at 10.
    seed : int
        Seed for the SimHash rotations (master seed for stacked stacks).
    center : bool, default=True
        Subtract the corpus mean from corpus and queries before encoding.
        Pure SimHash assumes mean-zero data — its sign-bit boundary is at
        the origin. Real embeddings (SPECTER2 has one dim with mean ≈ 15.5)
        violate this and SimHash collapses. Lloyd-Max 1-bit boundaries are
        adaptive (per-dim trained), so they implicitly center; the
        Lloyd-Max-equivalent SimHash is therefore SimHash on
        ``X - corpus.mean(0)``. The float32 ground truth is computed on the
        *raw* corpus regardless of this flag, so centering changes only the
        encoder's predictions, not the metric.

    Returns
    -------
    dict
        Row in the BASELINE.md table:
        ``{"dataset": str, "n": int, "d": int,
            "1-bit": float, "k=2": float, "k=4": float, "k=8": float}``
    """
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape}")
    n_total, d = int(emb.shape[0]), int(emb.shape[1])
    if d % 8 != 0:
        raise ValueError(
            f"d={d} not divisible by 8; remax codes are bit-packed bytewise"
        )

    corpus, queries = _split_queries(emb, n_queries, QUERY_SPLIT_SEED)
    truth = exact_knn(corpus, queries, k=k_eval)

    # Centering is applied to the encoder inputs only. Truth stays on raw
    # vectors so the metric is unchanged across the centered/uncentered
    # comparison and across the ladder.
    if center:
        mu = corpus.mean(axis=0)
        corpus_enc = corpus - mu
        queries_enc = queries - mu
    else:
        corpus_enc = corpus
        queries_enc = queries

    row: dict = {"dataset": name, "n": int(corpus.shape[0]), "d": d}

    # 1-bit — use Corpus to prove the metadata layer works end-to-end.
    with tempfile.TemporaryDirectory() as tmpdir:
        ids = [str(i) for i in range(corpus_enc.shape[0])]
        c = Corpus.build(tmpdir, corpus_enc, ids, d=d, seed=seed)
        pred_ids = []
        for q_vec in queries_enc:
            results = c.search(q_vec, k=k_eval)
            pred_ids.append([int(r.record_id) for r in results])
        pred_arr = np.array(pred_ids, dtype=np.intp)
    row["1-bit"] = recall_at_k(pred_arr, truth, k=k_eval)

    # Stacked ladder
    for k in LADDER_KS:
        qk = StackedSignBitQuantizer(d=d, k=k, seed=seed)
        resk = evaluate_quantizer(
            qk, corpus_enc, queries_enc, k_eval=k_eval, truth=truth
        )
        row[f"k={k}"] = resk["recall_at_k"]

    return row


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


def _fmt_recall(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def format_baseline_md(
    *,
    rows: Sequence[Mapping],
    version: str,
    n_queries: int,
    k_eval: int,
    seed: int,
    center: bool = True,
) -> str:
    """Render rows into the BASELINE.md table mandated by issue #4.

    The first column is the dataset name; remaining columns are ``n``, ``d``,
    ``1-bit``, ``k=2``, ``k=4``, ``k=8`` — order frozen by the issue.

    A protocol block at the top documents the query split, eval cutoff,
    seeds, and centering so the table is reproducible.
    """
    lines: list[str] = []
    lines.append(f"## remax v{version} baseline — R@{k_eval} vs float32")
    lines.append("")
    lines.append(f"- **Library version**: remax v{version}")
    lines.append(
        f"- **Eval metric**: R@{k_eval} vs float32 inner-product ground "
        f"truth (computed on raw, un-centered vectors)."
    )
    lines.append(
        f"- **Protocol**: {n_queries} held-out queries per dataset, "
        f"corpus = remainder. Query split seed = {QUERY_SPLIT_SEED}, "
        f"quantizer seed = {seed}."
    )
    if center:
        lines.append(
            "- **Centering**: corpus and queries are centered by the "
            "corpus mean before encoding. Pure SimHash assumes mean-zero "
            "data; real embeddings (SPECTER2 has one dim with mean ≈ 15.5) "
            "violate this. Lloyd-Max 1-bit boundaries are adaptive per "
            "dimension, so they implicitly center; the SimHash-equivalent "
            "is `sign(X - corpus.mean(0))`. Disable with `--no-center`."
        )
    else:
        lines.append(
            "- **Centering**: disabled (`--no-center`). Numbers are pure "
            "SimHash on raw embeddings; expect collapse on un-centered "
            "real data."
        )
    lines.append(
        "- **Hardware**: pure NumPy on CPU. No SIMD/Numba/GPU paths "
        "(those are post-v0.1.0 by design — see CLAUDE.md anti-goals)."
    )
    lines.append("")
    lines.append("| dataset       | n      | d   | 1-bit | k=2   | k=4   | k=8   |")
    lines.append("|---------------|--------|-----|-------|-------|-------|-------|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['dataset']:<13s}",
                    f"{r['n']:>6d}" if r.get("n") is not None else "—",
                    f"{r['d']:>3d}" if r.get("d") is not None else "—",
                    _fmt_recall(r.get("1-bit")),
                    _fmt_recall(r.get("k=2")),
                    _fmt_recall(r.get("k=4")),
                    _fmt_recall(r.get("k=8")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------- #


def _read_version() -> str:
    """Best-effort read of the package version from pyproject.toml.

    Falls back to ``remax.__version__`` (currently ``0.0.0``) if the file is
    not reachable from the installed package — fine for development checkouts.
    """
    try:
        import remax  # local import to avoid circular at module load
        return remax.__version__
    except Exception:
        return "0.0.0"


def _resolve_results_path(out: Optional[str]) -> Path:
    """Resolve the BASELINE.md output path.

    Default: ``<repo_root>/bench/results/BASELINE.md``. The function locates
    the in-repo ``bench/results/`` by walking up from this file, so it works
    both from a source checkout and from an installed editable package.
    """
    if out is not None:
        return Path(out).resolve()
    # src/remax/bench/run_baseline.py → repo_root/bench/results/BASELINE.md
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "bench" / "results" / "BASELINE.md"


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-baseline",
        description="Produce remax v0.1.0 baseline numbers (BASELINE.md).",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"max embeddings per dataset (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--queries", type=int, default=DEFAULT_N_QUERIES,
        help=f"held-out queries per dataset (default: {DEFAULT_N_QUERIES})",
    )
    p.add_argument(
        "--k-eval", type=int, default=DEFAULT_K_EVAL,
        help=f"R@K cutoff (default: {DEFAULT_K_EVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"quantizer master seed (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="output path for BASELINE.md (default: bench/results/BASELINE.md)",
    )
    p.add_argument(
        "--no-center", action="store_true",
        help=(
            "skip centering corpus/queries by corpus mean before encoding. "
            "Default behavior centers because pure SimHash assumes "
            "mean-zero data; real embeddings (e.g. SPECTER2) violate this "
            "and uncentered SimHash collapses. See compute_baseline_for_"
            "embeddings docstring."
        ),
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help=(
            "subset of dataset names to run (default: all registered). "
            "Unknown names are an error."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    target_datasets = (
        list(datasets.available_datasets())
        if args.datasets is None
        else list(args.datasets)
    )
    for name in target_datasets:
        # Validate names early so a typo at the end of a long run is caught
        # before we encode anything.
        datasets.dataset_spec(name)

    rows: list[dict] = []
    for name in target_datasets:
        try:
            X, info = datasets.load_dataset(name, n=args.n)
        except FileNotFoundError as e:
            sys.stderr.write(f"[skip] {name}: {e}\n\n")
            rows.append({
                "dataset": name,
                "n": None, "d": None,
                "1-bit": None, "k=2": None, "k=4": None, "k=8": None,
            })
            continue

        sys.stderr.write(f"[run]  {name}: n={info['n']} d={info['dim']}\n")
        row = compute_baseline_for_embeddings(
            name=name, emb=X,
            n_queries=args.queries, k_eval=args.k_eval, seed=args.seed,
            center=not args.no_center,
        )
        sys.stderr.write(
            f"       1-bit={row['1-bit']:.3f}  "
            f"k=2={row['k=2']:.3f}  k=4={row['k=4']:.3f}  "
            f"k=8={row['k=8']:.3f}\n"
        )
        rows.append(row)

    md = format_baseline_md(
        rows=rows,
        version=_read_version(),
        n_queries=args.queries,
        k_eval=args.k_eval,
        seed=args.seed,
        center=not args.no_center,
    )

    out_path = _resolve_results_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"\nwrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
