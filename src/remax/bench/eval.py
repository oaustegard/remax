"""Recall@K and float32 ground-truth helpers for the bench harness.

Ported from `remex/bench/onebit_experiment.py` (with input validation tightened)
so that the same primitive that produced the blog-post number lives here too.

Functions
---------
* :func:`recall_at_k` — fraction of true top-k overlapping predicted top-k,
  averaged over queries.
* :func:`exact_knn` — float32 inner-product ground truth.
* :func:`evaluate_quantizer` — drive a fitted quantizer over a corpus + query
  set and return one row of measurements.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["recall_at_k", "exact_knn", "evaluate_quantizer"]


# --------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------- #


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of true top-k indices found in predicted top-k, averaged over
    queries.

    Parameters
    ----------
    pred, truth : np.ndarray, shape (n_queries, ≥ k)
        Index arrays. Only the first ``k`` columns of each are consulted.
    k : int
        Cutoff. Must be positive and ``≤ min(pred.shape[1], truth.shape[1])``.

    Returns
    -------
    float
        ``mean_q |topk(pred[q]) ∩ topk(truth[q])| / k`` in ``[0, 1]``.
    """
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if pred.shape[0] != truth.shape[0]:
        raise ValueError(
            f"pred and truth must have the same number of queries; "
            f"got pred.shape={pred.shape}, truth.shape={truth.shape}."
        )
    if k > pred.shape[1] or k > truth.shape[1]:
        raise ValueError(
            f"k={k} exceeds available columns "
            f"(pred.shape[1]={pred.shape[1]}, truth.shape[1]={truth.shape[1]})."
        )
    hits = sum(
        len(set(p[:k].tolist()) & set(t[:k].tolist()))
        for p, t in zip(pred, truth)
    )
    return hits / (pred.shape[0] * k)


# --------------------------------------------------------------------- #
# exact_knn
# --------------------------------------------------------------------- #


def exact_knn(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Float32 inner-product top-k ground truth.

    Returns ``(n_queries, k)`` indices into ``corpus``, ordered by descending
    inner product. For unit-normed inputs this is cosine similarity.
    """
    corpus = np.asarray(corpus)
    queries = np.asarray(queries)
    if corpus.ndim != 2 or queries.ndim != 2:
        raise ValueError(
            f"corpus and queries must be 2-D; got "
            f"corpus.shape={corpus.shape}, queries.shape={queries.shape}."
        )
    if corpus.shape[1] != queries.shape[1]:
        raise ValueError(
            f"corpus and queries dim mismatch: "
            f"corpus.shape[1]={corpus.shape[1]}, "
            f"queries.shape[1]={queries.shape[1]}."
        )
    if k <= 0 or k > corpus.shape[0]:
        raise ValueError(
            f"k={k} out of range for corpus size {corpus.shape[0]}."
        )
    scores = queries @ corpus.T
    return np.argsort(-scores, axis=1)[:, :k]


# --------------------------------------------------------------------- #
# evaluate_quantizer
# --------------------------------------------------------------------- #


def evaluate_quantizer(
    quantizer: Any,
    corpus: np.ndarray,
    queries: np.ndarray,
    k_eval: int,
    truth: np.ndarray,
) -> Mapping[str, Any]:
    """Encode ``corpus``, run top-k Hamming search for each query, score
    against ``truth``, and return a metadata-rich result row.

    Parameters
    ----------
    quantizer : SignBitQuantizer | StackedSignBitQuantizer
        Fitted (or fit-on-construction) remax quantizer.
    corpus : (n, d) np.ndarray
    queries : (m, d) np.ndarray
    k_eval : int
        Top-k cutoff for both the search and the recall computation.
    truth : (m, ≥ k_eval) np.ndarray
        Ground-truth indices, typically from :func:`exact_knn`.

    Returns
    -------
    dict
        Keys: ``recall_at_k``, ``encoder``, ``n_bits``, optional ``k`` for
        stacked encoders.
    """
    codes = quantizer.encode(corpus)
    pred = quantizer.search(queries, codes, k=k_eval)
    if pred.ndim == 1:
        pred = pred[None, :]
    r = recall_at_k(pred, truth, k=k_eval)

    out: dict[str, Any] = {
        "recall_at_k": float(r),
        "encoder": type(quantizer).__name__,
        "n_bits": int(getattr(quantizer, "n_bits")),
    }
    if hasattr(quantizer, "k"):
        out["k"] = int(quantizer.k)
    return out
