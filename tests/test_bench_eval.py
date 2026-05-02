"""Tests for ``remax.bench.eval`` — recall@K computation and float32 ground truth.

Required by issue #4:

  1. ``recall_at_k(pred, truth, k)`` — fraction of true top-k found in
     predicted top-k, averaged over queries.
  2. ``exact_knn(corpus, queries, k)`` — float32 inner-product ground truth.
  3. ``evaluate_quantizer(quantizer, corpus, queries, k_eval, truth)`` —
     run a fitted quantizer over a corpus + query set, return R@K.

These primitives carry the contract that the blog-post number depends on, so
they get full unit coverage before the run_baseline orchestrator is wired up.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax import SignBitQuantizer, StackedSignBitQuantizer
from remax.bench.eval import (
    exact_knn,
    recall_at_k,
    evaluate_quantizer,
)


# --------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------- #


def test_recall_at_k_perfect_match():
    """Identical pred and truth → recall == 1.0 at any k."""
    truth = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    pred = truth.copy()
    assert recall_at_k(pred, truth, k=4) == pytest.approx(1.0)
    assert recall_at_k(pred, truth, k=2) == pytest.approx(1.0)
    assert recall_at_k(pred, truth, k=1) == pytest.approx(1.0)


def test_recall_at_k_no_overlap():
    """Disjoint pred and truth → recall == 0.0."""
    truth = np.array([[0, 1, 2, 3]])
    pred = np.array([[4, 5, 6, 7]])
    assert recall_at_k(pred, truth, k=4) == pytest.approx(0.0)


def test_recall_at_k_partial_overlap():
    """Half overlap on a single query → recall == 0.5 at k=4."""
    truth = np.array([[0, 1, 2, 3]])
    pred = np.array([[0, 1, 8, 9]])  # 2 of top-4 match
    assert recall_at_k(pred, truth, k=4) == pytest.approx(0.5)


def test_recall_at_k_set_semantics_ignores_order():
    """Recall is set-based — order within top-k must not matter."""
    truth = np.array([[0, 1, 2, 3]])
    pred_same_set = np.array([[3, 2, 1, 0]])
    assert recall_at_k(pred_same_set, truth, k=4) == pytest.approx(1.0)


def test_recall_at_k_averages_across_queries():
    """Per-query recalls should be averaged, not summed."""
    truth = np.array([[0, 1], [2, 3]])
    # Query 0: 1.0 (full match). Query 1: 0.0 (no match).
    pred = np.array([[0, 1], [9, 8]])
    assert recall_at_k(pred, truth, k=2) == pytest.approx(0.5)


def test_recall_at_k_truncates_to_k():
    """Columns beyond k must not contaminate the score."""
    truth = np.array([[0, 1, 2, 3, 4]])
    pred = np.array([[0, 1, 99, 98, 97]])  # top-2 match, rest random
    assert recall_at_k(pred, truth, k=2) == pytest.approx(1.0)
    assert recall_at_k(pred, truth, k=5) == pytest.approx(2.0 / 5.0)


def test_recall_at_k_shape_mismatch_raises():
    """pred and truth with different leading shape → ValueError."""
    truth = np.array([[0, 1, 2]])
    pred = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError):
        recall_at_k(pred, truth, k=3)


def test_recall_at_k_k_too_large_raises():
    """k larger than min(pred.shape[1], truth.shape[1]) → ValueError."""
    truth = np.array([[0, 1, 2]])
    pred = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        recall_at_k(pred, truth, k=4)


def test_recall_at_k_nonpositive_k_raises():
    truth = np.array([[0, 1]])
    pred = np.array([[0, 1]])
    with pytest.raises(ValueError):
        recall_at_k(pred, truth, k=0)
    with pytest.raises(ValueError):
        recall_at_k(pred, truth, k=-1)


# --------------------------------------------------------------------- #
# exact_knn
# --------------------------------------------------------------------- #


def test_exact_knn_self_search_returns_self_first():
    """A query identical to a corpus row must rank that row first."""
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((50, 16)).astype(np.float32)
    # Query vector is corpus[7] exactly
    queries = corpus[[7, 23]]
    top = exact_knn(corpus, queries, k=1)
    assert top.shape == (2, 1)
    assert top[0, 0] == 7
    assert top[1, 0] == 23


def test_exact_knn_shape_and_dtype():
    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((100, 32)).astype(np.float32)
    queries = rng.standard_normal((10, 32)).astype(np.float32)
    top = exact_knn(corpus, queries, k=5)
    assert top.shape == (10, 5)
    assert np.issubdtype(top.dtype, np.integer)
    # All indices must be in [0, n_corpus)
    assert top.min() >= 0
    assert top.max() < 100


def test_exact_knn_matches_argsort_inner_product():
    """Reference implementation: argsort of -X @ Y.T."""
    rng = np.random.default_rng(2)
    corpus = rng.standard_normal((40, 8)).astype(np.float32)
    queries = rng.standard_normal((3, 8)).astype(np.float32)
    expected = np.argsort(-(queries @ corpus.T), axis=1)[:, :7]
    got = exact_knn(corpus, queries, k=7)
    np.testing.assert_array_equal(got, expected)


def test_exact_knn_k_larger_than_corpus_raises():
    rng = np.random.default_rng(3)
    corpus = rng.standard_normal((10, 4)).astype(np.float32)
    queries = rng.standard_normal((2, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        exact_knn(corpus, queries, k=11)


def test_exact_knn_dim_mismatch_raises():
    rng = np.random.default_rng(4)
    corpus = rng.standard_normal((10, 4)).astype(np.float32)
    queries = rng.standard_normal((2, 5)).astype(np.float32)
    with pytest.raises(ValueError):
        exact_knn(corpus, queries, k=3)


# --------------------------------------------------------------------- #
# evaluate_quantizer
# --------------------------------------------------------------------- #


def test_evaluate_quantizer_returns_recall_and_meta_for_signbit():
    """1-bit quantizer on an easy synthetic problem should give R@10 well above
    chance, and the metadata block must report the encoder name + n_bits.
    """
    rng = np.random.default_rng(0)
    n, d = 500, 64
    X = rng.standard_normal((n, d)).astype(np.float32)
    # Carve a held-out query set (10 of them)
    queries = X[:10].copy()
    corpus = X[10:].copy()
    truth = exact_knn(corpus, queries, k=10)

    q = SignBitQuantizer(d=d, seed=42)
    result = evaluate_quantizer(q, corpus, queries, k_eval=10, truth=truth)

    assert "recall_at_k" in result
    assert isinstance(result["recall_at_k"], float)
    assert 0.0 <= result["recall_at_k"] <= 1.0
    # 1-bit on isotropic Gaussian at d=64, n=490 corpus easily clears chance (~0.02)
    assert result["recall_at_k"] > 0.10
    # Metadata for the markdown table
    assert result["n_bits"] == d
    assert result["encoder"] == "SignBitQuantizer"


def test_evaluate_quantizer_returns_recall_and_meta_for_stacked():
    """Stacked quantizer must report total bits = k*d and encoder name."""
    rng = np.random.default_rng(0)
    n, d = 300, 32
    X = rng.standard_normal((n, d)).astype(np.float32)
    queries = X[:8].copy()
    corpus = X[8:].copy()
    truth = exact_knn(corpus, queries, k=10)

    q = StackedSignBitQuantizer(d=d, k=4, seed=42)
    result = evaluate_quantizer(q, corpus, queries, k_eval=10, truth=truth)
    assert result["encoder"] == "StackedSignBitQuantizer"
    assert result["n_bits"] == 4 * d
    assert "k" in result
    assert result["k"] == 4


def test_evaluate_quantizer_more_stacks_higher_recall():
    """Variance shrinks ∝ 1/k. k=8 should beat k=1 (in expectation, on enough
    queries) for a non-trivial recall problem."""
    rng = np.random.default_rng(0)
    n, d = 800, 64
    X = rng.standard_normal((n, d)).astype(np.float32)
    queries = X[:30].copy()
    corpus = X[30:].copy()
    truth = exact_knn(corpus, queries, k=10)

    r_k1 = evaluate_quantizer(
        StackedSignBitQuantizer(d=d, k=1, seed=42),
        corpus, queries, k_eval=10, truth=truth,
    )["recall_at_k"]
    r_k8 = evaluate_quantizer(
        StackedSignBitQuantizer(d=d, k=8, seed=42),
        corpus, queries, k_eval=10, truth=truth,
    )["recall_at_k"]
    # Strict inequality on 30 queries is the contract — if it ever fails it's
    # diagnostic of a real regression.
    assert r_k8 > r_k1
