"""remax.characterize — encoder characterization utility.

Sweeps a strategy × k grid on a user-supplied corpus/query set and reports
the recommended operating point for sign-bit compression.

Detects:
- Whether embeddings are L2-normalized (norms ≈ 1.0)
- Whether centering helps or hurts at the recommended k
- Whether PCA is worth reaching for at extreme compression
- Matryoshka training floor (if any) via graceful-degradation curve

Most of the sweep logic originates in ``bench/sketch_matryoshka.py``; this
module wraps it as a library function with a clean, programmatic output format.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["characterize", "CharacterizeReport"]

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

DEFAULT_STRATEGIES: list[str] = ["sign-raw", "sign-centered", "pca", "haar-trunc"]
DEFAULT_K_VALUES: list[int] = [64, 128, 256, 384, 512, 768]

VALID_STRATEGIES: frozenset[str] = frozenset(
    [
        "sign-raw",
        "sign-centered",
        "pca",
        "haar-trunc",
        "gaussian",
        "countsketch",
        "f32-raw",
        "f32-centered",
    ]
)

_TRUTH_K = 10  # ground-truth cutoff when computed internally


# ── Low-level search primitives ───────────────────────────────────────────────


def _sign_pack(X: np.ndarray) -> np.ndarray:
    """sign(X) → packed uint8, padding trailing dim to a multiple of 8."""
    if X.ndim == 1:
        X = X[None, :]
    pad = (8 - X.shape[1] % 8) % 8
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)))
    return np.packbits(X > 0, axis=1)


def _hamming_topN(q_codes: np.ndarray, c_codes: np.ndarray, N: int) -> np.ndarray:
    """Top-N by Hamming distance, returns (nq, min(N, n)) index array."""
    N = min(N, c_codes.shape[0])
    nq = q_codes.shape[0]
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        d = _POPCOUNT_LUT[np.bitwise_xor(c_codes, q_codes[i])].sum(1)
        part = np.argpartition(d, N)[:N]
        out[i] = part[np.argsort(d[part])]
    return out


def _float32_topN(queries: np.ndarray, corpus: np.ndarray, N: int) -> np.ndarray:
    """Top-N by float32 IP, returns (nq, min(N, n)) index array."""
    N = min(N, corpus.shape[0])
    sims = queries @ corpus.T
    nq = queries.shape[0]
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        part = np.argpartition(-sims[i], N)[:N]
        out[i] = part[np.argsort(-sims[i, part])]
    return out


def _recall_set(truth: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of items in *truth* found anywhere in *pred*, mean over queries."""
    k = truth.shape[1]
    hits = sum(
        len(set(truth[i].tolist()) & set(pred[i].tolist()))
        for i in range(truth.shape[0])
    )
    return hits / (truth.shape[0] * k)


# ── Report types ──────────────────────────────────────────────────────────────


@dataclass
class CharacterizeReport:
    """Result of :func:`characterize`.

    Attributes
    ----------
    best : dict
        Recommended operating point for a 1-bit strategy. Keys:
        ``strategy``, ``k``, ``R@10``, ``R@100``, ``B_vec``
        (bytes per vector at that k).
    table : list[dict]
        Full strategy × k grid. Each row dict has the same keys as
        ``best``.
    notes : str
        Human-readable observations: L2-normalization status, centering
        effect, PCA benefit at extreme compression, and Matryoshka floor
        (if detectable from the float32 degradation curve).

    Metrics
    -------
    R@10 : fraction of true top-10 found in the predicted top-10.
    R@100 : fraction of true top-10 found in the predicted top-100.
    (When ``ground_truth`` is provided its width replaces 10 in the above.)
    """

    best: dict
    table: list[dict]
    notes: str

    def __str__(self) -> str:
        lines = [f"Best: {self.best}", ""]
        lines.append(
            f"{'strategy':<16} {'k':>6} {'R@10':>7} {'R@100':>7} {'B/vec':>6}"
        )
        lines.append("─" * 50)
        prev_k = None
        for row in self.table:
            if prev_k is not None and row["k"] != prev_k:
                lines.append("")
            lines.append(
                f"{row['strategy']:<16} {row['k']:>6} {row['R@10']:>7.3f} "
                f"{row['R@100']:>7.3f} {row['B_vec']:>6}"
            )
            prev_k = row["k"]
        if self.notes:
            lines += ["", "Notes:"]
            for note in self.notes.splitlines():
                lines.append(f"  {note}")
        return "\n".join(lines)


# ── Main API ──────────────────────────────────────────────────────────────────


def characterize(
    corpus: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray | None = None,
    *,
    strategies: Sequence[str] | None = None,
    k_values: Sequence[int] | None = None,
    seed: int = 99,
    verbose: bool = False,
) -> CharacterizeReport:
    """Characterize sign-bit compression strategies for an embedding encoder.

    Sweeps the requested strategy × k grid and returns the recommended
    operating point along with a full results table and geometry notes.

    Parameters
    ----------
    corpus : np.ndarray, shape (N, D)
        Corpus embeddings.
    queries : np.ndarray, shape (Q, D)
        Query embeddings.  Must share dimension D with ``corpus``.
    ground_truth : np.ndarray, shape (Q, K), optional
        Pre-computed ground-truth top-K indices from a full float32 IP
        search.  If omitted, the top-10 by float32 IP is computed
        internally from the provided corpus and queries.
    strategies : list[str], optional
        Subset of ``{"sign-raw", "sign-centered", "pca", "haar-trunc",
        "gaussian", "countsketch", "f32-raw", "f32-centered"}``.
        Default: ``["sign-raw", "sign-centered", "pca", "haar-trunc"]``.
    k_values : list[int], optional
        Dimensionality (bit-budget) values to test.  Values exceeding D
        are silently dropped.
        Default: ``[64, 128, 256, 384, 512, 768]`` clipped to D.
    seed : int
        RNG seed for random projections (``gaussian``, ``countsketch``)
        and Haar rotation.
    verbose : bool
        Print progress messages to stderr.

    Returns
    -------
    CharacterizeReport
        ``.best`` — recommended operating point dict.
        ``.table`` — full strategy × k grid as a list of dicts.
        ``.notes`` — human-readable geometry observations.

    Examples
    --------
    >>> import numpy as np
    >>> from remax import characterize
    >>> rng = np.random.default_rng(0)
    >>> corpus = rng.standard_normal((500, 128)).astype(np.float32)
    >>> queries = rng.standard_normal((20, 128)).astype(np.float32)
    >>> report = characterize(corpus, queries, k_values=[64, 128])
    >>> report.best["strategy"] in {"sign-raw", "sign-centered", "pca", "haar-trunc"}
    True
    """
    corpus = np.asarray(corpus, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    if corpus.ndim != 2:
        raise ValueError(f"corpus must be 2-D, got ndim={corpus.ndim}")
    if queries.ndim != 2:
        raise ValueError(f"queries must be 2-D, got ndim={queries.ndim}")
    if corpus.shape[1] != queries.shape[1]:
        raise ValueError(
            f"corpus dim {corpus.shape[1]} != queries dim {queries.shape[1]}"
        )

    d = corpus.shape[1]

    # Strategy list
    if strategies is None:
        active_strategies = list(DEFAULT_STRATEGIES)
    else:
        active_strategies = list(strategies)
    unknown = set(active_strategies) - VALID_STRATEGIES
    if unknown:
        raise ValueError(
            f"Unknown strategies: {unknown!r}. Valid: {sorted(VALID_STRATEGIES)}"
        )

    # k values
    if k_values is None:
        active_ks = [k for k in DEFAULT_K_VALUES if k <= d]
    else:
        active_ks = [k for k in k_values if k <= d]
    if not active_ks:
        raise ValueError(
            f"No valid k values: all exceed corpus dimension d={d}."
        )

    # Ground truth (top-K indices)
    if ground_truth is None:
        if verbose:
            print("# Computing float32 ground truth (top-10)…", file=sys.stderr)
        truth = _float32_topN(queries, corpus, _TRUTH_K)
    else:
        truth = np.asarray(ground_truth, dtype=np.intp)
        if truth.ndim != 2 or truth.shape[0] != queries.shape[0]:
            raise ValueError(
                f"ground_truth must be (Q={queries.shape[0]}, K) array, "
                f"got {truth.shape}"
            )

    # ── Precomputes ───────────────────────────────────────────────────────

    # Corpus geometry
    norms_sample = np.linalg.norm(corpus[: min(200, len(corpus))].astype(np.float64), axis=1)
    is_normalized = bool(
        abs(norms_sample.mean() - 1.0) < 0.05 and norms_sample.std() < 0.05
    )

    mu = corpus.mean(0)
    corpus_c = (corpus - mu)
    queries_c = (queries - mu)

    # PCA (SVD on centered corpus)
    Vt: np.ndarray | None = None
    if "pca" in active_strategies:
        if verbose:
            print("# Precomputing PCA (SVD)…", file=sys.stderr)
        _, _, Vt = np.linalg.svd(corpus_c.astype(np.float64), full_matrices=False)
        if verbose:
            print("# SVD done.", file=sys.stderr)

    # Haar rotation
    H: np.ndarray | None = None
    if "haar-trunc" in active_strategies:
        from .rotation import haar_rotation

        if verbose:
            print(f"# Precomputing Haar rotation at d={d}…", file=sys.stderr)
        H = haar_rotation(d, seed=seed).astype(np.float32)
        if verbose:
            print("# Haar done.", file=sys.stderr)

    # Gaussian projection matrix (per-k so we seed consistently and build lazily)
    # Count-sketch projection matrix (per-k likewise)

    # ── Strategy × k sweep ───────────────────────────────────────────────

    table: list[dict] = []

    for k in active_ks:
        for strategy in active_strategies:
            if strategy == "f32-raw":
                c_data = corpus[:, :k].astype(np.float32)
                q_data = queries[:, :k].astype(np.float32)
                pred = _float32_topN(q_data, c_data, 100)
                bpv = k * 4

            elif strategy == "f32-centered":
                c_data = corpus_c[:, :k].astype(np.float32)
                q_data = queries_c[:, :k].astype(np.float32)
                pred = _float32_topN(q_data, c_data, 100)
                bpv = k * 4

            elif strategy == "sign-raw":
                cc = _sign_pack(corpus[:, :k])
                qc = _sign_pack(queries[:, :k])
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            elif strategy == "sign-centered":
                cc = _sign_pack(corpus_c[:, :k])
                qc = _sign_pack(queries_c[:, :k])
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            elif strategy == "pca":
                assert Vt is not None
                cc = _sign_pack(corpus_c.astype(np.float64) @ Vt[:k].T)
                qc = _sign_pack(queries_c.astype(np.float64) @ Vt[:k].T)
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            elif strategy == "haar-trunc":
                assert H is not None
                cc = _sign_pack((corpus_c @ H)[:, :k])
                qc = _sign_pack((queries_c @ H)[:, :k])
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            elif strategy == "gaussian":
                gr = np.random.default_rng(seed + 1)
                R = gr.standard_normal((d, k)).astype(np.float32) / np.sqrt(k)
                cc = _sign_pack(corpus_c @ R)
                qc = _sign_pack(queries_c @ R)
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            elif strategy == "countsketch":
                cr = np.random.default_rng(seed + 2)
                buckets = cr.integers(0, k, size=d)
                signs = cr.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)
                Sk = np.zeros((d, k), dtype=np.float32)
                Sk[np.arange(d), buckets] = signs
                cc = _sign_pack(corpus_c @ Sk)
                qc = _sign_pack(queries_c @ Sk)
                pred = _hamming_topN(qc, cc, 100)
                bpv = cc.shape[1]

            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")  # unreachable

            r10 = _recall_set(truth, pred[:, :10])
            r100 = _recall_set(truth, pred)

            table.append(
                {
                    "strategy": strategy,
                    "k": k,
                    "R@10": round(r10, 4),
                    "R@100": round(r100, 4),
                    "B_vec": bpv,
                }
            )

    # ── Best operating point ──────────────────────────────────────────────
    # Prefer 1-bit strategies (highest R@100, ties broken by R@10).
    onebit = [r for r in table if not r["strategy"].startswith("f32-")]
    pool = onebit if onebit else table
    best_row = max(pool, key=lambda r: (r["R@100"], r["R@10"]))
    best = dict(best_row)

    # ── Notes ─────────────────────────────────────────────────────────────
    notes_parts: list[str] = []

    if is_normalized:
        notes_parts.append("L2-normalized: all corpus norms ≈ 1.0.")
    else:
        notes_parts.append(
            f"Not L2-normalized: corpus norms mean={norms_sample.mean():.2f}, "
            f"std={norms_sample.std():.2f}."
        )

    best_k = best_row["k"]

    def _r100(strategy: str, k: int) -> float | None:
        for r in table:
            if r["strategy"] == strategy and r["k"] == k:
                return r["R@100"]
        return None

    if "sign-raw" in active_strategies and "sign-centered" in active_strategies:
        raw = _r100("sign-raw", best_k)
        cen = _r100("sign-centered", best_k)
        if raw is not None and cen is not None:
            delta = cen - raw
            if delta > 0.005:
                notes_parts.append(
                    f"Centering helps at k={best_k}: "
                    f"R@100 {raw:.3f} → {cen:.3f} (+{delta:.3f})."
                )
            elif delta < -0.005:
                notes_parts.append(
                    f"Centering hurts at k={best_k}: "
                    f"R@100 {raw:.3f} → {cen:.3f} ({delta:.3f})."
                )
            else:
                notes_parts.append(
                    f"Centering has negligible effect at k={best_k}."
                )

    if (
        "pca" in active_strategies
        and "sign-centered" in active_strategies
        and active_ks
    ):
        low_k = active_ks[0]
        pca_r = _r100("pca", low_k)
        sc_r = _r100("sign-centered", low_k)
        if pca_r is not None and sc_r is not None:
            delta = pca_r - sc_r
            if delta > 0.01:
                notes_parts.append(
                    f"PCA worth reaching for at k={low_k}: "
                    f"R@100 {sc_r:.3f} → {pca_r:.3f} (+{delta:.3f})."
                )
            else:
                notes_parts.append(
                    f"PCA offers no benefit over centering at k={low_k}."
                )

    # Matryoshka floor: sharp drop in f32 quality below some k
    f32_strat = (
        "f32-centered"
        if "f32-centered" in active_strategies
        else ("f32-raw" if "f32-raw" in active_strategies else None)
    )
    if f32_strat and len(active_ks) >= 2:
        f32_rows = sorted(
            [r for r in table if r["strategy"] == f32_strat],
            key=lambda r: r["k"],
        )
        if f32_rows:
            max_r100 = f32_rows[-1]["R@100"]
            floor_k = None
            for row in reversed(f32_rows):
                if row["R@100"] < max_r100 * 0.9:
                    floor_k = row["k"]
                    break
            if floor_k is not None:
                notes_parts.append(
                    f"Matryoshka floor: {f32_strat} quality drops sharply below k={floor_k}."
                )

    notes = "\n".join(notes_parts)

    return CharacterizeReport(best=best, table=table, notes=notes)
