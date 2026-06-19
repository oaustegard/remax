"""Tests for rank-correctness metrics (issue #46).

:mod:`remax.bench.ranking` — Kendall τ-b and cosine-graded nDCG between a
Hamming ranking and the full-precision cosine ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax.bench.ranking import (
    kendall_tau,
    mean_kendall_tau,
    mean_ndcg_at_k,
    ndcg_at_k,
)


# --------------------------------------------------------------------- #
# kendall_tau
# --------------------------------------------------------------------- #
def test_kendall_tau_perfect_agreement():
    """Hamming order exactly matches cosine order → τ = +1.

    Higher cosine ⇔ lower Hamming distance.
    """
    cos = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    ham = np.array([1, 2, 3, 4, 5])  # nearer (smaller) ↔ higher cosine
    assert kendall_tau(ham, cos) == pytest.approx(1.0)


def test_kendall_tau_reversed():
    cos = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    ham = np.array([1, 2, 3, 4, 5][::-1])
    assert kendall_tau(ham, cos) == pytest.approx(-1.0)


def test_kendall_tau_constant_is_nan():
    cos = np.array([0.5, 0.5, 0.5, 0.5])
    ham = np.array([1, 2, 3, 4])
    assert np.isnan(kendall_tau(ham, cos))


def test_kendall_tau_shape_mismatch_raises():
    with pytest.raises(ValueError):
        kendall_tau(np.array([1, 2, 3]), np.array([0.1, 0.2]))


# --------------------------------------------------------------------- #
# ndcg_at_k
# --------------------------------------------------------------------- #
def test_ndcg_perfect_ordering_is_one():
    cos = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    pred = np.array([0, 1, 2, 3, 4])  # already in ideal (descending) order
    assert ndcg_at_k(pred, cos, k=3) == pytest.approx(1.0)


def test_ndcg_suboptimal_below_one():
    cos = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    pred = np.array([4, 3, 2, 1, 0])  # worst-first
    assert ndcg_at_k(pred, cos, k=3) < 1.0


def test_ndcg_all_negative_cosine_is_zero():
    """No item has positive cosine → ideal DCG is 0 → nDCG defined as 0."""
    cos = np.array([-0.2, -0.5, -0.1, -0.9])
    pred = np.array([0, 2, 1, 3])
    assert ndcg_at_k(pred, cos, k=2) == 0.0


def test_ndcg_k_validation():
    cos = np.array([0.9, 0.7, 0.5])
    with pytest.raises(ValueError):
        ndcg_at_k(np.array([0, 1, 2]), cos, k=0)
    with pytest.raises(ValueError):
        ndcg_at_k(np.array([0, 1]), cos, k=3)  # k exceeds pred length


# --------------------------------------------------------------------- #
# mean wrappers
# --------------------------------------------------------------------- #
def test_mean_kendall_tau_drops_nan_rows():
    cos = np.array([[0.9, 0.7, 0.5], [0.5, 0.5, 0.5]])  # row 1 constant → nan
    ham = np.array([[1, 2, 3], [1, 2, 3]])
    # only the first (perfect) row counts → mean τ = 1.0
    assert mean_kendall_tau(ham, cos) == pytest.approx(1.0)


def test_mean_ndcg_averages():
    cos = np.array([[0.9, 0.7, 0.5], [0.9, 0.7, 0.5]])
    pred = np.array([[0, 1, 2], [2, 1, 0]])
    v = mean_ndcg_at_k(pred, cos, k=2)
    assert 0.0 < v < 1.0


def test_mean_ndcg_query_count_mismatch_raises():
    with pytest.raises(ValueError):
        mean_ndcg_at_k(np.zeros((2, 3), dtype=int), np.zeros((3, 3)), k=2)
