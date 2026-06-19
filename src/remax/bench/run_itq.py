"""Issue #46 spike — learned hash projections (ITQ) vs centered SimHash.

Compares the parameter-free centered-SimHash ladder
(:class:`remax.StackedSignBitQuantizer`) against learned ITQ rotations
(:class:`remax.StackedITQQuantizer`) at **equal bit budget**, on real SPECTER2
embeddings, reporting both set-membership recall and rank-correctness.

Three encoders per eval set, at each ladder rung ``k ∈ {1, 2, 4, 8}``
(``k·d`` bits):

* ``haar``      — centered SimHash baseline (data-agnostic Haar rotations).
* ``itq_in``    — ITQ rotations learned on the eval set's own corpus.
* ``itq_xfer``  — ITQ rotations learned on a *different* corpus (the
  cross-corpus transfer probe — the make-or-break for learned hashing).

Metrics, all vs full-precision float32 cosine:

* ``R@k``    — Recall@k_eval (set overlap with the float32 top-k).
* ``tau``    — Kendall τ-b of the Hamming order vs the cosine order over the
  whole corpus (rank-correctness).
* ``nDCG@k`` — cosine-graded nDCG@k_eval of the Hamming ranking.

Flavor (b) — relevance-distilled projections under teacher (Claude) triplet
labels — is **out of scope here** and documented as such: the issue gates it on
flavor (a) landing *and* teacher relevance labels being available, and the
cached SPECTER2 corpora carry no such labels. This runner is flavor (a).

Usage
-----
::

    bash bench/fetch_specter2_cache.sh                 # broad → SPECTER2/
    # narrow corpus for the transfer probe: fetch specter2_nlp_narrow.npy into
    #   bench/.cache/SPECTER2_NARROW/embeddings.npy   (see issue #46 thread)
    python -m remax.bench.run_itq                       # writes bench/results/ITQ.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from remax import StackedITQQuantizer, StackedSignBitQuantizer
from remax.bench import datasets
from remax.bench.eval import exact_knn, recall_at_k
from remax.bench.ranking import mean_kendall_tau, mean_ndcg_at_k
from remax.packing import hamming_distances, stable_top_k

__all__ = [
    "LADDER_KS",
    "compute_itq_experiment",
    "format_itq_md",
    "main",
]

# Ladder rungs as bit budgets ×d. k=1 is the single-rotation "1-bit" rung; the
# issue's ladder-rung breakdown wants the learned-vs-random delta at each.
LADDER_KS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_N = 10_000
DEFAULT_N_QUERIES = 100
DEFAULT_K_EVAL = 10
DEFAULT_SEED = 42
DEFAULT_ITQ_ITERS = 50
# Mirror run_baseline so the query split lands identically across experiments.
QUERY_SPLIT_SEED = 99


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _split_queries(
    emb: np.ndarray, n_queries: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(corpus, queries) split — same permutation protocol as run_baseline."""
    if n_queries <= 0 or n_queries >= emb.shape[0]:
        raise ValueError(
            f"n_queries={n_queries} out of range for n={emb.shape[0]}"
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(emb.shape[0])
    return emb[perm[n_queries:]], emb[perm[:n_queries]]


def _l2norm(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows pass through unchanged."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return X / n


def _itq_prefix(full: StackedITQQuantizer, k: int) -> StackedITQQuantizer:
    """A k-rung view of a fitted max-k ITQ quantiser, sharing its first k
    rotations and training mean.

    SeedSequence is prefix-nested (``generate_state(k)`` is the head of
    ``generate_state(k_max)``), so the first ``k`` rotations of a k_max fit are
    byte-identical to an independent k fit — this slicing is exact, not an
    approximation, and saves re-running ITQ per rung.
    """
    q = StackedITQQuantizer(
        d=full.d, k=k, seed=full.seed, n_iters=full.n_iters, dtype=full.dtype
    )
    q.rotations_ = full.rotations_[:k].copy()
    q.mean_ = full.mean_
    return q


def _measure(
    quantizer,
    corpus_enc: np.ndarray,
    queries_enc: np.ndarray,
    truth: np.ndarray,
    cos: np.ndarray,
    k_eval: int,
) -> dict:
    """One (recall, tau, nDCG) measurement for a fitted/seeded quantiser.

    ``corpus_enc`` / ``queries_enc`` are the *raw* (un-centered) rows for ITQ
    — it centers internally — and the *pre-centered* rows for Haar (which does
    not center). ``cos`` is the (m, n) full-precision cosine matrix used as the
    rank-correctness reference; ``truth`` the float32 top-k_eval indices.
    """
    c_codes = quantizer.encode(corpus_enc)
    q_codes = quantizer.encode(queries_enc)
    if q_codes.ndim == 1:
        q_codes = q_codes[None, :]
    m = q_codes.shape[0]

    hd = np.empty((m, corpus_enc.shape[0]), dtype=np.int64)
    pred = np.empty((m, k_eval), dtype=np.intp)
    for i in range(m):
        di = hamming_distances(c_codes, q_codes[i])
        hd[i] = di
        pred[i] = stable_top_k(di, k_eval)

    return {
        "recall": recall_at_k(pred, truth, k=k_eval),
        "tau": mean_kendall_tau(hd, cos),
        "ndcg": mean_ndcg_at_k(pred, cos, k_eval),
        "n_bits": int(quantizer.n_bits),
    }


# --------------------------------------------------------------------- #
# Core experiment
# --------------------------------------------------------------------- #


def compute_itq_experiment(
    *,
    eval_name: str,
    eval_emb: np.ndarray,
    transfer_name: str,
    transfer_emb: np.ndarray,
    ladder_ks: Sequence[int] = LADDER_KS,
    n_queries: int = DEFAULT_N_QUERIES,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    itq_iters: int = DEFAULT_ITQ_ITERS,
) -> dict:
    """Run the haar / itq_in / itq_xfer comparison on one eval set.

    Parameters
    ----------
    eval_name, eval_emb
        The dataset retrieval is measured on. Split into (corpus, queries) via
        ``QUERY_SPLIT_SEED``. ``itq_in`` learns on its corpus.
    transfer_name, transfer_emb
        The *other* corpus whose ITQ rotation is applied to ``eval_emb`` —
        the cross-corpus transfer probe. ``transfer_emb`` is used corpus-side
        (its own query split removed, to keep the fit on index-like rows).
    ladder_ks
        Rungs to evaluate. Must be ascending; ITQ is fit once at the max rung
        and sliced (see :func:`_itq_prefix`).
    seed, itq_iters
        Quantiser master seed and ITQ iteration count.

    Returns
    -------
    dict
        ``{"eval": str, "transfer": str, "n": int, "d": int, "k_eval": int,
            "rows": [ {k, n_bits, haar:{...}, itq_in:{...}, itq_xfer:{...}}, ...]}``
    """
    eval_emb = np.asarray(eval_emb, dtype=np.float32)
    transfer_emb = np.asarray(transfer_emb, dtype=np.float32)
    if eval_emb.ndim != 2:
        raise ValueError(f"eval_emb must be 2-D, got {eval_emb.shape}")
    d = int(eval_emb.shape[1])
    if d % 8 != 0:
        raise ValueError(f"d={d} not divisible by 8 (bit-packed codes)")
    if transfer_emb.shape[1] != d:
        raise ValueError(
            f"transfer dim {transfer_emb.shape[1]} != eval dim {d}; a learned "
            "rotation is d×d and cannot transfer across dimensions."
        )
    ladder_ks = list(ladder_ks)
    if ladder_ks != sorted(ladder_ks) or ladder_ks[0] <= 0:
        raise ValueError(f"ladder_ks must be ascending positive ints: {ladder_ks}")
    k_max = ladder_ks[-1]

    corpus, queries = _split_queries(eval_emb, n_queries, QUERY_SPLIT_SEED)
    # Transfer fit uses the foreign corpus's index-like rows (drop its queries
    # so the comparison "rotation learned elsewhere" is corpus-to-corpus).
    transfer_corpus, _ = _split_queries(transfer_emb, n_queries, QUERY_SPLIT_SEED)

    # Ground truth + cosine reference, both on RAW eval vectors (centering is
    # an encoder concern; the metric is fixed across methods).
    truth = exact_knn(corpus, queries, k=k_eval)
    cos = _l2norm(queries) @ _l2norm(corpus).T  # (m, n) cosine relevance

    # Pre-centered views for Haar (it does not center); ITQ gets raw rows.
    mu = corpus.mean(axis=0)
    corpus_c = corpus - mu
    queries_c = queries - mu

    # Fit ITQ once at the max rung for each fit-corpus, then slice per rung.
    itq_in_full = StackedITQQuantizer(
        d=d, k=k_max, seed=seed, n_iters=itq_iters
    ).fit(corpus)
    itq_xfer_full = StackedITQQuantizer(
        d=d, k=k_max, seed=seed, n_iters=itq_iters
    ).fit(transfer_corpus)

    rows: list[dict] = []
    for k in ladder_ks:
        haar = StackedSignBitQuantizer(d=d, k=k, seed=seed)
        row = {
            "k": k,
            "n_bits": k * d,
            "haar": _measure(haar, corpus_c, queries_c, truth, cos, k_eval),
            "itq_in": _measure(
                _itq_prefix(itq_in_full, k), corpus, queries, truth, cos, k_eval
            ),
            "itq_xfer": _measure(
                _itq_prefix(itq_xfer_full, k), corpus, queries, truth, cos, k_eval
            ),
        }
        rows.append(row)

    return {
        "eval": eval_name,
        "transfer": transfer_name,
        "n": int(corpus.shape[0]),
        "d": d,
        "k_eval": k_eval,
        "rows": rows,
    }


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


def _f(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}"


def _delta(a: float, b: float) -> str:
    """Signed delta a−b, with sign, for the transfer/win columns."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return "—"
    return f"{a - b:+.3f}"


def _result_table(res: Mapping) -> list[str]:
    """One eval-set block: per-rung R@k / tau / nDCG for the three encoders."""
    ke = res["k_eval"]
    lines: list[str] = []
    lines.append(
        f"### eval = {res['eval']}  (transfer rotation from {res['transfer']})"
    )
    lines.append("")
    lines.append(f"- n={res['n']}, d={res['d']}, R@{ke}, τ = Kendall τ-b vs cosine order.")
    lines.append(
        f"- **win** = `itq_in R@{ke} − haar R@{ke}` at equal bits. "
        f"**xfer Δ** = `itq_xfer R@{ke} − itq_in R@{ke}` (transfer penalty)."
    )
    lines.append("")
    header = (
        f"| k | bits | haar R@{ke} | itq_in R@{ke} | itq_xfer R@{ke} "
        f"| win | xfer Δ | haar τ | itq_in τ | itq_xfer τ "
        f"| haar nDCG | itq_in nDCG | itq_xfer nDCG |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines.append(header)
    lines.append(sep)
    for r in res["rows"]:
        h, ii, ix = r["haar"], r["itq_in"], r["itq_xfer"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['k']}",
                    f"{r['n_bits']}",
                    _f(h["recall"]),
                    _f(ii["recall"]),
                    _f(ix["recall"]),
                    _delta(ii["recall"], h["recall"]),
                    _delta(ix["recall"], ii["recall"]),
                    _f(h["tau"]),
                    _f(ii["tau"]),
                    _f(ix["tau"]),
                    _f(h["ndcg"]),
                    _f(ii["ndcg"]),
                    _f(ix["ndcg"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def format_itq_md(
    *,
    results: Sequence[Mapping],
    version: str,
    n_queries: int,
    seed: int,
    itq_iters: int,
) -> str:
    """Render the ITQ.md report from one or more :func:`compute_itq_experiment`
    results."""
    lines: list[str] = []
    lines.append("## Issue #46 — learned hash projections (ITQ) vs centered SimHash")
    lines.append("")
    lines.append(f"- **Library**: remax v{version}, pure NumPy/SciPy on CPU.")
    lines.append(
        f"- **Protocol**: {n_queries} held-out queries per eval set "
        f"(split seed {QUERY_SPLIT_SEED}), corpus = remainder. Quantiser seed "
        f"{seed}; ITQ {itq_iters} iters/rotation."
    )
    lines.append(
        "- **Encoders**: `haar` = centered SimHash (Haar rotations, "
        "data-agnostic); `itq_in` = ITQ learned on the eval corpus; "
        "`itq_xfer` = ITQ learned on the *other* corpus (cross-corpus probe)."
    )
    lines.append(
        "- **Metrics vs float32 cosine**: R@k (top-k set overlap), Kendall "
        "τ-b (full-corpus rank agreement), cosine-graded nDCG@k. Ground truth "
        "and cosine reference are on raw vectors; centering is encoder-side only."
    )
    lines.append(
        "- **Equal bits**: every rung compares encoders at the same `k·d` bit "
        "budget. Flavor (b) relevance-distilled projections are out of scope "
        "(no teacher labels in these corpora; gated on flavor (a) — see runner "
        "docstring)."
    )
    lines.append("")
    for res in results:
        lines += _result_table(res)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _read_version() -> str:
    try:
        import remax
        return remax.__version__
    except Exception:
        return "0.0.0"


def _resolve_results_path(out: Optional[str]) -> Path:
    if out is not None:
        return Path(out).resolve()
    here = Path(__file__).resolve()
    return here.parents[3] / "bench" / "results" / "ITQ.md"


def _load_npy_dataset(name: str, cache_subdir: str, n: Optional[int]) -> np.ndarray:
    """Load embeddings from ``bench/.cache/<cache_subdir>/embeddings.npy``.

    SPECTER2 is registered in :mod:`remax.bench.datasets`; the narrow transfer
    corpus is not (it shares SPECTER2's dim and encoder), so it is loaded by
    path. Both go through the same shape validation.
    """
    if cache_subdir == "SPECTER2":
        emb, _info = datasets.load_dataset("SPECTER2", n=n)
        return emb
    root = datasets._CACHE_ROOT  # respects REMAX_BENCH_CACHE_DIR override
    path = Path(root) / cache_subdir / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {name} embeddings at {path}\n"
            f"to fix: fetch specter2_nlp_narrow.npy from "
            f"oaustegard/claude-container-layers@specter2-nlp-narrow-10k into "
            f"{path} (see issue #46 thread / bench/fetch_specter2_cache.sh)"
        )
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 768:
        raise ValueError(f"{name} at {path} has shape {arr.shape}; expected (*, 768)")
    if n is not None and n < arr.shape[0]:
        arr = arr[:n]
    return np.ascontiguousarray(arr, dtype=np.float32)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-itq",
        description="Issue #46: learned ITQ rotations vs centered SimHash.",
    )
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--queries", type=int, default=DEFAULT_N_QUERIES)
    p.add_argument("--k-eval", type=int, default=DEFAULT_K_EVAL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--itq-iters", type=int, default=DEFAULT_ITQ_ITERS)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--broad-subdir", type=str, default="SPECTER2",
        help="cache subdir for the broad corpus (default: SPECTER2)",
    )
    p.add_argument(
        "--narrow-subdir", type=str, default="SPECTER2_NARROW",
        help="cache subdir for the narrow transfer corpus",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    broad = _load_npy_dataset("SPECTER2-broad", args.broad_subdir, args.n)
    narrow = _load_npy_dataset("SPECTER2-narrow", args.narrow_subdir, args.n)
    sys.stderr.write(
        f"[load] broad={broad.shape} narrow={narrow.shape}\n"
    )

    results = []
    for eval_name, eval_emb, xfer_name, xfer_emb in [
        ("SPECTER2-broad", broad, "SPECTER2-narrow", narrow),
        ("SPECTER2-narrow", narrow, "SPECTER2-broad", broad),
    ]:
        sys.stderr.write(f"[run] eval={eval_name} transfer-from={xfer_name}\n")
        res = compute_itq_experiment(
            eval_name=eval_name, eval_emb=eval_emb,
            transfer_name=xfer_name, transfer_emb=xfer_emb,
            n_queries=args.queries, k_eval=args.k_eval,
            seed=args.seed, itq_iters=args.itq_iters,
        )
        for r in res["rows"]:
            sys.stderr.write(
                f"      k={r['k']:>1}  haar R={r['haar']['recall']:.3f}  "
                f"itq_in R={r['itq_in']['recall']:.3f}  "
                f"itq_xfer R={r['itq_xfer']['recall']:.3f}\n"
            )
        results.append(res)

    md = format_itq_md(
        results=results, version=_read_version(),
        n_queries=args.queries, seed=args.seed, itq_iters=args.itq_iters,
    )
    out_path = _resolve_results_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"\nwrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
