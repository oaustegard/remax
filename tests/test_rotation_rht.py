"""Tests for the randomized Hadamard rotation and the shared rotation surface.

Three groups:

1. **Mechanics** — orthogonality, determinism, dtype, argument validation, and
   the ``rotation=`` plumbing on both quantizer classes.
2. **The Charikar guarantee** — the empirical collision rate must track
   ``θ/π``. This is the property the RHT could plausibly have broken, since a
   randomized Hadamard is not Haar-distributed; it is asserted directly
   against the published curve rather than against Haar's behaviour, because
   two equally-structured rotations agreeing with each other would prove
   nothing (remax#59).
3. **Stack independence** — the assumption ``StackedSignBitQuantizer`` rests
   on. This is the one that actually failed for the naive single-round
   construction, so it is guarded explicitly rather than assumed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from remax import (SignBitQuantizer, StackedSignBitQuantizer, haar_rotation,
                   rht_rotation)
from remax.rotation import ROTATIONS, build_rotation

ROT_NAMES = sorted(ROTATIONS)


# --------------------------------------------------------------------- #
# 1. Mechanics
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [8, 64, 128, 192, 512, 768])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rht_is_orthogonal(d, dtype):
    """Covers powers of two (single-block FWHT) and non-powers (d=192, 768)."""
    R = rht_rotation(d, seed=0, dtype=dtype)
    assert R.shape == (d, d)
    assert R.dtype == np.dtype(dtype)
    atol = 8 * np.sqrt(d) * np.finfo(dtype).eps
    np.testing.assert_allclose(R @ R.T, np.eye(d, dtype=dtype), atol=atol)
    np.testing.assert_allclose(R.T @ R, np.eye(d, dtype=dtype), atol=atol)


def test_rht_preserves_norms_and_angles():
    """An orthogonal map is exactly what SimHash needs: angles must survive."""
    d = 256
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, d)).astype(np.float32)
    R = rht_rotation(d, seed=1)
    XR = X @ R
    np.testing.assert_allclose(
        np.linalg.norm(XR, axis=1), np.linalg.norm(X, axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(XR @ XR.T, X @ X.T, atol=1e-3)


def test_rht_seed_determinism():
    a = rht_rotation(128, seed=7)
    b = rht_rotation(128, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, rht_rotation(128, seed=8))


def test_rht_differs_from_haar():
    assert not np.allclose(rht_rotation(64, seed=3), haar_rotation(64, seed=3))


def test_rht_rejects_odd_d():
    with pytest.raises(ValueError, match="odd"):
        rht_rotation(63, seed=0)


@pytest.mark.parametrize("d", [0, -8, 1.5])
def test_rht_rejects_bad_d(d):
    with pytest.raises(ValueError, match="positive integer"):
        rht_rotation(d, seed=0)


@pytest.mark.parametrize("rounds", [1, 0, -1])
def test_rht_rejects_single_round(rounds):
    """The floor exists for a measured reason; keep the door shut.

    A single mixing round leaves the k rotations of a stack correlated on
    anisotropic input — see ``test_rht_stacks_stay_independent`` for the
    property that breaks, and remax#59 for the numbers.
    """
    with pytest.raises(ValueError, match="below the minimum"):
        rht_rotation(64, seed=0, rounds=rounds)


def test_rht_extra_rounds_are_accepted_and_orthogonal():
    R = rht_rotation(128, seed=0, rounds=5)
    np.testing.assert_allclose(R @ R.T, np.eye(128, dtype=np.float32), atol=1e-5)
    assert not np.allclose(R, rht_rotation(128, seed=0, rounds=2))


@pytest.mark.parametrize("kind", ROT_NAMES)
def test_build_rotation_dispatch(kind):
    np.testing.assert_array_equal(
        build_rotation(kind, 64, seed=5), ROTATIONS[kind](64, seed=5)
    )


@pytest.mark.parametrize("bad", ["hadamard", "", None, 3])
def test_build_rotation_rejects_unknown(bad):
    with pytest.raises(ValueError, match="unknown rotation"):
        build_rotation(bad, 64, seed=0)


# --------------------------------------------------------------------- #
# 1b. Quantizer plumbing
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ROT_NAMES)
def test_quantizers_accept_rotation_kind(kind):
    q = SignBitQuantizer(d=64, seed=1, rotation=kind)
    assert q.rotation == kind
    np.testing.assert_array_equal(q.rotation_, ROTATIONS[kind](64, seed=1))

    s = StackedSignBitQuantizer(d=64, k=3, seed=1, rotation=kind)
    assert s.rotation == kind
    assert s.rotations_.shape == (3, 64, 64)


def test_quantizer_rotation_defaults_to_haar():
    assert SignBitQuantizer(d=64, seed=1).rotation == "haar"
    assert StackedSignBitQuantizer(d=64, k=2, seed=1).rotation == "haar"
    # The default must stay bit-identical to the pre-#59 behaviour: an
    # existing encoded corpus has to keep searching correctly.
    np.testing.assert_array_equal(
        SignBitQuantizer(d=64, seed=1).rotation_, haar_rotation(64, seed=1)
    )


@pytest.mark.parametrize("cls,kwargs", [
    (SignBitQuantizer, {"d": 64, "seed": 0}),
    (StackedSignBitQuantizer, {"d": 64, "k": 2, "seed": 0}),
])
def test_quantizers_reject_unknown_rotation(cls, kwargs):
    with pytest.raises(ValueError, match="unknown rotation"):
        cls(rotation="ffht", **kwargs)


@pytest.mark.parametrize("kind", ROT_NAMES)
def test_rht_stacked_roundtrip(kind):
    """Self-search must return self at distance 0 under either rotation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 128)).astype(np.float32)
    q = StackedSignBitQuantizer(d=128, k=4, seed=9, rotation=kind)
    codes = q.encode(X)
    assert codes.shape == (200, 4 * 128 // 8)
    idx, dist = q.search(X[3], codes, k=5, return_distances=True)
    assert idx[0] == 3
    assert dist[0] == 0


def test_rht_and_haar_produce_different_codes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 128)).astype(np.float32)
    a = StackedSignBitQuantizer(d=128, k=2, seed=5, rotation="haar").encode(X)
    b = StackedSignBitQuantizer(d=128, k=2, seed=5, rotation="rht").encode(X)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------- #
# 1c. rotations_ is a view, not a second copy
# --------------------------------------------------------------------- #
def test_rotations_view_shares_storage_with_projection_matrix():
    """``rotations_`` must not duplicate the stack in memory.

    Storing it separately doubled resident rotation bytes (604 MB at
    d=3072, k=8) and left two representations that could desync.
    """
    q = StackedSignBitQuantizer(d=64, k=4, seed=2)
    assert not q.rotations_.flags.owndata
    assert np.shares_memory(q.rotations_, q._rotation_matrix)
    assert "rotations_" not in q.__dict__


def test_rotations_view_matches_projection_matrix_blocks():
    d, k = 64, 4
    q = StackedSignBitQuantizer(d=d, k=k, seed=2)
    for j in range(k):
        np.testing.assert_array_equal(
            q.rotations_[j], q._rotation_matrix[:, j * d : (j + 1) * d]
        )


def test_rotations_view_writes_propagate():
    """Writing through the view reaches the matmul buffer — no silent desync."""
    q = StackedSignBitQuantizer(d=64, k=2, seed=2)
    q.rotations_[1, 0, 0] = 12.5
    assert q._rotation_matrix[0, 64] == 12.5


# --------------------------------------------------------------------- #
# 2. The Charikar guarantee — collision rate vs the published curve
# --------------------------------------------------------------------- #
def _pairs_at_angle(theta, n_pairs, d, rng, anisotropic=False):
    """Unit vectors with exactly the requested pairwise angle."""
    U = rng.standard_normal((n_pairs, d))
    V = rng.standard_normal((n_pairs, d))
    if anisotropic:
        # Axis-aligned variance decay plus a shared mean direction. Real
        # embeddings look like this (the "rogue dimension" phenomenon), and
        # it is the regime where a structured rotation is most exposed.
        scale = 1.0 / np.sqrt(np.arange(1, d + 1))
        mean_dir = rng.standard_normal(d)
        mean_dir /= np.linalg.norm(mean_dir)
        U = U * scale + 3.0 * mean_dir
        V = V * scale + 3.0 * mean_dir
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V -= np.sum(V * U, axis=1, keepdims=True) * U
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Y = np.cos(theta) * U + np.sin(theta) * V
    return U.astype(np.float32), Y.astype(np.float32)


def _collision_fracs(X, Y, q):
    """Per-stack disagreement fractions, shape (n_pairs, k)."""
    d, k = q.d, q.k
    sx = (X @ q._rotation_matrix) > 0
    sy = (Y @ q._rotation_matrix) > 0
    diff = (sx != sy).reshape(X.shape[0], k, d)
    return diff.mean(axis=2)


@pytest.mark.parametrize("kind", ROT_NAMES)
@pytest.mark.parametrize("frac", [0.1, 0.25, 0.5, 0.75])
def test_collision_rate_tracks_theta_over_pi(kind, frac):
    """P[sign mismatch] must equal θ/π — the guarantee remax rests on.

    Asserted against the published curve, not against the other rotation.
    """
    d, k, n_pairs = 256, 8, 400
    theta = math.pi * frac
    rng = np.random.default_rng(0xC0FFEE)
    X, Y = _pairs_at_angle(theta, n_pairs, d, rng)
    q = StackedSignBitQuantizer(d=d, k=k, seed=17, rotation=kind)
    observed = _collision_fracs(X, Y, q).mean()
    # Standard error of the mean over n_pairs draws is ~sqrt(p(1-p)/(k*d*n));
    # 0.006 is a wide multiple of that, so this is a guard, not a coin flip.
    assert abs(observed - frac) < 0.006, (
        f"{kind}: collision rate {observed:.4f} vs theory {frac:.4f}"
    )


@pytest.mark.parametrize("kind", ROT_NAMES)
def test_collision_rate_holds_on_anisotropic_input(kind):
    """Same claim, on the input distribution that breaks a naive RHT."""
    d, k, n_pairs, frac = 256, 8, 400, 0.5
    rng = np.random.default_rng(0xBEEF)
    X, Y = _pairs_at_angle(math.pi * frac, n_pairs, d, rng, anisotropic=True)
    q = StackedSignBitQuantizer(d=d, k=k, seed=23, rotation=kind)
    observed = _collision_fracs(X, Y, q).mean()
    assert abs(observed - frac) < 0.008, (
        f"{kind}: collision rate {observed:.4f} vs theory {frac:.4f}"
    )


# --------------------------------------------------------------------- #
# 3. Stack independence — the assumption the ladder rests on
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ROT_NAMES)
def test_stacks_stay_independent_on_anisotropic_input(kind):
    """Pooling k stacks must actually deliver the 1/k variance reduction.

    ``SeedSequence`` gives independent *seeds*, but a structured transform can
    still yield correlated hyperplane sets — seeds are not the assumption,
    independence of the resulting estimators is. Measured on the naive
    single-round RHT at d=512, k=8: ratio 1.77 and mean |corr| 0.11, i.e. a
    k=8 stack delivering the variance of k≈4.5. At two rounds both land at
    Haar's level (0.96 / 0.04), which is what this asserts.
    """
    d, k, n_pairs = 512, 8, 400
    rng = np.random.default_rng(0xFEED)
    X, Y = _pairs_at_angle(math.pi * 0.5, n_pairs, d, rng, anisotropic=True)
    q = StackedSignBitQuantizer(d=d, k=k, seed=31, rotation=kind)
    fr = _collision_fracs(X, Y, q)

    var_ratio = fr.mean(axis=1).var() / (fr.var(axis=0).mean() / k)
    corr = np.corrcoef(fr.T)
    mean_abs_corr = np.abs(corr[~np.eye(k, dtype=bool)]).mean()

    assert var_ratio < 1.30, (
        f"{kind}: pooled variance is {var_ratio:.2f}x what independent stacks "
        f"would give — the k-stack ladder is not delivering 1/k"
    )
    assert mean_abs_corr < 0.07, (
        f"{kind}: mean |cross-stack correlation| {mean_abs_corr:.3f} is above "
        f"the ~0.04 sampling floor"
    )


@pytest.mark.parametrize("kind", ROT_NAMES)
def test_stacking_shrinks_estimator_spread(kind):
    """The ladder's headline claim: spread falls as 1/sqrt(k)."""
    d, n_pairs = 256, 500
    rng = np.random.default_rng(0xD00D)
    X, Y = _pairs_at_angle(math.pi * 0.5, n_pairs, d, rng)
    sds = []
    for k in (1, 4, 16):
        q = StackedSignBitQuantizer(d=d, k=k, seed=41, rotation=kind)
        sds.append(_collision_fracs(X, Y, q).mean(axis=1).std())
    # Each 4x in k should roughly halve the spread. Allow generous slack;
    # the point is monotone shrinkage at about the right rate.
    assert sds[1] < sds[0] * 0.75, f"{kind}: k=4 spread {sds} did not shrink"
    assert sds[2] < sds[1] * 0.75, f"{kind}: k=16 spread {sds} did not shrink"
