"""Rank-correctness metrics for Hamming-vs-cosine ranking agreement.

``recall_at_k`` (in :mod:`remax.bench.eval`) answers "did the right items land
in the top-k set". These metrics answer the finer question issue #46 asks of a
*learned* code: **how faithfully does the Hamming ordering reproduce the
full-precision cosine ordering**, not just the top-k membership.

* :func:`kendall_tau` — Kendall τ-b between the cosine similarities and the
  (negated) Hamming distances over the *whole* candidate list, for one query.
  τ-b is the tie-aware variant — essential because integer Hamming distances
  tie heavily. +1 = identical order, 0 = independent, −1 = reversed.
* :func:`ndcg_at_k` — normalised DCG of the Hamming-induced top-k ordering,
  graded by cosine relevance. Rewards putting high-cosine items *early*, not
  merely *present* (which recall measures).
* :func:`mean_kendall_tau` / :func:`mean_ndcg_at_k` — query-averaged wrappers.

Cosine gains
------------
``ndcg_at_k`` grades each item by ``max(0, cosine)``. Inputs are assumed
L2-normalised by the caller so inner product == cosine ∈ [−1, 1]; negative-cosine
items are treated as irrelevant (gain 0). This keeps DCG non-negative and the
normalisation well-defined.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

__all__ = [
    "kendall_tau",
    "ndcg_at_k",
    "mean_kendall_tau",
    "mean_ndcg_at_k",
]


def kendall_tau(hamming_dists: np.ndarray, cosine_sims: np.ndarray) -> float:
    """Kendall τ-b between cosine order and Hamming order, for one query.

    Parameters
    ----------
    hamming_dists : np.ndarray, shape (n,)
        Hamming distance from the query to every corpus item (smaller = nearer).
    cosine_sims : np.ndarray, shape (n,)
        Full-precision cosine similarity to every corpus item (larger = nearer).

    Returns
    -------
    float
        τ-b in ``[-1, 1]``. Hamming distance is negated so that a *perfect*
        code (Hamming order == cosine order) yields ``+1``. Returns ``nan`` if
        either side is constant (τ undefined), matching scipy.
    """
    hamming_dists = np.asarray(hamming_dists, dtype=np.float64).ravel()
    cosine_sims = np.asarray(cosine_sims, dtype=np.float64).ravel()
    if hamming_dists.shape != cosine_sims.shape:
        raise ValueError(
            f"shape mismatch: hamming_dists {hamming_dists.shape} vs "
            f"cosine_sims {cosine_sims.shape}"
        )
    tau, _p = kendalltau(cosine_sims, -hamming_dists, variant="b")
    return float(tau)


def ndcg_at_k(pred_order: np.ndarray, cosine_sims: np.ndarray, k: int) -> float:
    """Cosine-graded nDCG@k of a Hamming-induced ranking, for one query.

    Parameters
    ----------
    pred_order : np.ndarray, shape (>= k,)
        Corpus indices ranked best-first by the code (e.g. ascending Hamming
        distance). Only the first ``k`` are scored.
    cosine_sims : np.ndarray, shape (n,)
        Full-precision cosine similarity for every corpus item. Used both as
        the graded relevance (``gain = max(0, cosine)``) and to build the
        ideal ranking.
    k : int
        Cutoff. Must be positive and ``<= len(pred_order)``.

    Returns
    -------
    float
        ``DCG@k(pred) / DCG@k(ideal)`` in ``[0, 1]``. ``0.0`` when the ideal
        DCG is zero (no item has positive cosine — degenerate query).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pred_order = np.asarray(pred_order).ravel()
    if k > pred_order.shape[0]:
        raise ValueError(
            f"k={k} exceeds pred_order length {pred_order.shape[0]}"
        )
    cosine_sims = np.asarray(cosine_sims, dtype=np.float64).ravel()
    gains_all = np.maximum(0.0, cosine_sims)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    pred_gains = gains_all[pred_order[:k]]
    dcg = float((pred_gains * discounts).sum())

    ideal_gains = np.sort(gains_all)[::-1][:k]
    idcg = float((ideal_gains * discounts).sum())
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_kendall_tau(
    hamming_dists: np.ndarray, cosine_sims: np.ndarray
) -> float:
    """Mean per-query :func:`kendall_tau` over a ``(m, n)`` batch.

    NaN τ values (constant rows) are dropped from the average; if every row is
    degenerate the result is ``nan``.
    """
    hamming_dists = np.asarray(hamming_dists, dtype=np.float64)
    cosine_sims = np.asarray(cosine_sims, dtype=np.float64)
    if hamming_dists.shape != cosine_sims.shape or hamming_dists.ndim != 2:
        raise ValueError(
            "hamming_dists and cosine_sims must be matching 2-D (m, n) arrays; "
            f"got {hamming_dists.shape} and {cosine_sims.shape}"
        )
    taus = [
        kendall_tau(hamming_dists[i], cosine_sims[i])
        for i in range(hamming_dists.shape[0])
    ]
    taus = np.asarray(taus, dtype=np.float64)
    if np.all(np.isnan(taus)):
        return float("nan")
    return float(np.nanmean(taus))


def mean_ndcg_at_k(
    pred_orders: np.ndarray, cosine_sims: np.ndarray, k: int
) -> float:
    """Mean per-query :func:`ndcg_at_k` over a batch.

    Parameters
    ----------
    pred_orders : np.ndarray, shape (m, >= k)
        Per-query Hamming-induced rankings (best-first indices).
    cosine_sims : np.ndarray, shape (m, n)
        Per-query full-precision cosine similarities.
    k : int
        Cutoff.
    """
    pred_orders = np.asarray(pred_orders)
    cosine_sims = np.asarray(cosine_sims, dtype=np.float64)
    if pred_orders.ndim != 2 or cosine_sims.ndim != 2:
        raise ValueError(
            "pred_orders and cosine_sims must both be 2-D; got "
            f"{pred_orders.shape} and {cosine_sims.shape}"
        )
    if pred_orders.shape[0] != cosine_sims.shape[0]:
        raise ValueError(
            f"query-count mismatch: pred_orders {pred_orders.shape[0]} vs "
            f"cosine_sims {cosine_sims.shape[0]}"
        )
    vals = [
        ndcg_at_k(pred_orders[i], cosine_sims[i], k)
        for i in range(pred_orders.shape[0])
    ]
    return float(np.mean(vals))
