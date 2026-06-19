"""Tests for learned ITQ rotations (issue #46).

Covers the rotation primitive (:func:`remax.rotation.itq_rotation`) and the
quantiser (:class:`remax.itq.StackedITQQuantizer`):

  1. The learned rotation is orthogonal and deterministic under a fixed seed.
  2. ITQ does what it claims — the sign-quantisation loss is non-increasing
     across iterations and ends ≤ a Haar rotation's on anisotropic data.
  3. The quantiser refuses to encode before ``fit`` and round-trips after it.
  4. Encoded size, self-recall, and transfer-fit (fit on A, encode B) behave.
  5. SeedSequence prefix-nesting: the first k of a k_max fit equal a k fit
     — the property ``run_itq._itq_prefix`` relies on.

Synthetic data is an anisotropic Gaussian (correlated covariance) so the
learned rotation has structure to exploit; an isotropic Gaussian gives ITQ
nothing over Haar and would make the loss-reduction test vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax import StackedITQQuantizer, haar_rotation, itq_rotation


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _anisotropic(n: int, d: int, *, rng: np.random.Generator) -> np.ndarray:
    """Centered anisotropic Gaussian: correlated covariance so principal
    axes are not the coordinate axes (gives ITQ something to learn)."""
    A = rng.standard_normal((d, d))
    cov = A @ A.T
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    return (X - X.mean(0)).astype(np.float32)


def _sign_quant_loss(X: np.ndarray, R: np.ndarray) -> float:
    """‖sign(XR) − XR‖²_F — the ITQ objective."""
    Z = X @ R
    B = np.where(Z >= 0.0, 1.0, -1.0)
    return float(np.linalg.norm(B - Z) ** 2)


# --------------------------------------------------------------------- #
# 1. Rotation primitive
# --------------------------------------------------------------------- #
def test_itq_rotation_is_orthogonal():
    rng = np.random.default_rng(0)
    X = _anisotropic(2000, 64, rng=rng)
    R = itq_rotation(X, n_iters=30, seed=1)
    eye = np.eye(64)
    np.testing.assert_allclose(R @ R.T, eye, atol=1e-4)


def test_itq_rotation_deterministic_under_seed():
    rng = np.random.default_rng(0)
    X = _anisotropic(1500, 56, rng=rng)
    R1 = itq_rotation(X, n_iters=20, seed=7)
    R2 = itq_rotation(X, n_iters=20, seed=7)
    np.testing.assert_array_equal(R1, R2)


def test_itq_rotation_different_seeds_differ():
    rng = np.random.default_rng(0)
    X = _anisotropic(1500, 56, rng=rng)
    R1 = itq_rotation(X, n_iters=20, seed=1)
    R2 = itq_rotation(X, n_iters=20, seed=2)
    assert np.linalg.norm(R1 - R2) > 1e-3


def test_itq_loss_non_increasing_across_iterations():
    """The alternating ITQ update can only lower (or hold) the loss."""
    rng = np.random.default_rng(3)
    X = _anisotropic(3000, 64, rng=rng).astype(np.float64)
    losses = [
        _sign_quant_loss(X, itq_rotation(X, n_iters=it, seed=0))
        for it in (1, 2, 5, 10, 25, 50)
    ]
    for earlier, later in zip(losses, losses[1:]):
        assert later <= earlier + 1e-6, f"loss rose: {losses}"


def test_itq_beats_haar_quantization_loss():
    """At convergence ITQ's sign-quant loss is ≤ a random Haar rotation's."""
    rng = np.random.default_rng(4)
    X = _anisotropic(3000, 64, rng=rng).astype(np.float64)
    haar_loss = _sign_quant_loss(X, haar_rotation(64, seed=0, dtype=np.float64))
    itq_loss = _sign_quant_loss(X, itq_rotation(X, n_iters=50, seed=0))
    assert itq_loss <= haar_loss


def test_itq_rotation_rejects_bad_args():
    rng = np.random.default_rng(0)
    X = _anisotropic(100, 32, rng=rng)
    with pytest.raises(ValueError):
        itq_rotation(X[0], n_iters=10)          # 1-D
    with pytest.raises(ValueError):
        itq_rotation(X, n_iters=0)              # non-positive iters


# --------------------------------------------------------------------- #
# 2. Quantiser lifecycle
# --------------------------------------------------------------------- #
def test_encode_before_fit_raises():
    q = StackedITQQuantizer(d=64, k=2, seed=0)
    rng = np.random.default_rng(0)
    X = _anisotropic(500, 64, rng=rng)
    with pytest.raises(RuntimeError):
        q.encode(X)
    with pytest.raises(RuntimeError):
        q.search(X[0], np.zeros((10, 2 * 64 // 8), dtype=np.uint8))


def test_fit_sets_rotations_and_mean():
    rng = np.random.default_rng(0)
    X = _anisotropic(800, 64, rng=rng)
    q = StackedITQQuantizer(d=64, k=4, seed=1, n_iters=10).fit(X)
    assert q.rotations_.shape == (4, 64, 64)
    assert q.mean_.shape == (64,)
    # each rotation orthogonal
    for j in range(4):
        np.testing.assert_allclose(
            q.rotations_[j] @ q.rotations_[j].T, np.eye(64), atol=1e-4
        )


def test_encoded_size_and_self_recall():
    rng = np.random.default_rng(2)
    X = _anisotropic(1000, 64, rng=rng)
    q = StackedITQQuantizer(d=64, k=4, seed=1, n_iters=10).fit(X)
    codes = q.encode(X)
    assert codes.shape == (1000, 4 * 64 // 8)
    assert codes.dtype == np.uint8
    # a vector's nearest Hamming neighbour is itself
    top = q.search(X[0], codes, k=5)
    assert top[0] == 0


def test_single_vector_encode_shape():
    rng = np.random.default_rng(2)
    X = _anisotropic(400, 64, rng=rng)
    q = StackedITQQuantizer(d=64, k=2, seed=1, n_iters=5).fit(X)
    code = q.encode(X[0])
    assert code.shape == (2 * 64 // 8,)


def test_transfer_fit_then_encode_other_corpus():
    """Fit on corpus A, encode corpus B — the cross-corpus path. The training
    mean travels with the rotation (encode subtracts mean_, not B's mean)."""
    rng = np.random.default_rng(5)
    A = _anisotropic(1000, 64, rng=rng)
    B = _anisotropic(800, 64, rng=rng) + 3.0  # different mean and spread
    q = StackedITQQuantizer(d=64, k=2, seed=1, n_iters=10).fit(A)
    mean_before = q.mean_.copy()
    codes = q.encode(B)
    assert codes.shape == (800, 2 * 64 // 8)
    np.testing.assert_array_equal(q.mean_, mean_before)  # encode is read-only


def test_constructor_validation():
    with pytest.raises(ValueError):
        StackedITQQuantizer(d=63, k=2)          # not divisible by 8
    with pytest.raises(ValueError):
        StackedITQQuantizer(d=64, k=0)          # non-positive k
    with pytest.raises(ValueError):
        StackedITQQuantizer(d=64, k=2, n_iters=0)
    with pytest.raises(ValueError):
        StackedITQQuantizer(d=64, k=2).fit(np.zeros((10, 32)))  # wrong d


# --------------------------------------------------------------------- #
# 3. SeedSequence prefix-nesting (used by run_itq._itq_prefix)
# --------------------------------------------------------------------- #
def test_fit_is_prefix_nested_in_k():
    """First k rotations of a k_max fit are byte-identical to a k fit.

    This is what lets the experiment fit ITQ once at k=8 and slice the rungs.
    """
    rng = np.random.default_rng(6)
    X = _anisotropic(700, 64, rng=rng)
    q8 = StackedITQQuantizer(d=64, k=8, seed=11, n_iters=8).fit(X)
    q4 = StackedITQQuantizer(d=64, k=4, seed=11, n_iters=8).fit(X)
    np.testing.assert_array_equal(q8.rotations_[:4], q4.rotations_)
