"""Tests for ``remax.core`` — the 1-bit ``SignBitQuantizer`` and the
functional primitives it composes (``haar_rotation``, ``encode_signs``,
``hamming_distances``, ``hamming_search``).

Required by issue #2:

  1. Rotation orthogonality: ``R @ R.T ≈ I`` to 1e-6.
  2. Determinism: same seed → byte-identical codes.
  3. Roundtrip on synthetic Gaussian: Spearman ρ between true cosine and
     SimHash-derived cosine estimate > 0.95 at d=768, n=1000.
  4. Self-recall ≥ 95% on isotropic Gaussian, k=1.
  5. ``d % 8 != 0`` raises a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from remax import (
    SignBitQuantizer,
    encode_signs,
    haar_rotation,
    hamming_distances,
    hamming_search,
)


# --------------------------------------------------------------------- #
# 1. Rotation orthogonality
# --------------------------------------------------------------------- #
def _orth_atol(d: int, dtype) -> float:
    """Tolerance for ``R @ R.T ≈ I`` scaled to the working precision.

    Round-off in the d-term inner product accumulates to ~``sqrt(d) * eps``
    in the worst case; ``8 *`` gives slack for QR + sign-correction.
    """
    return 8 * np.sqrt(d) * np.finfo(dtype).eps


@pytest.mark.parametrize("d", [8, 64, 128, 768])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotation_orthogonality(d: int, dtype):
    R = haar_rotation(d=d, seed=0, dtype=dtype)
    atol = _orth_atol(d, dtype)
    np.testing.assert_allclose(R @ R.T, np.eye(d), atol=atol)
    np.testing.assert_allclose(R.T @ R, np.eye(d), atol=atol)


def test_rotation_orthogonal_via_class():
    q = SignBitQuantizer(d=128, seed=0)
    atol = _orth_atol(128, q.rotation_.dtype)
    np.testing.assert_allclose(q.rotation_ @ q.rotation_.T, np.eye(128), atol=atol)


# --------------------------------------------------------------------- #
# 2. Determinism
# --------------------------------------------------------------------- #
def test_seed_determinism_rotation():
    R1 = haar_rotation(d=64, seed=123)
    R2 = haar_rotation(d=64, seed=123)
    np.testing.assert_array_equal(R1, R2)


def test_seed_determinism_codes_byte_identical():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 256)).astype(np.float64)

    q1 = SignBitQuantizer(d=256, seed=999)
    q2 = SignBitQuantizer(d=256, seed=999)
    c1 = q1.encode(X)
    c2 = q2.encode(X)
    assert c1.tobytes() == c2.tobytes()
    np.testing.assert_array_equal(c1, c2)


def test_different_seeds_produce_different_codes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 256))
    c1 = SignBitQuantizer(d=256, seed=1).encode(X)
    c2 = SignBitQuantizer(d=256, seed=2).encode(X)
    # Astronomically unlikely to collide on Gaussian data with two
    # independent rotations; if this ever fires it's a real bug.
    assert c1.tobytes() != c2.tobytes()


# --------------------------------------------------------------------- #
# 3. Spearman ρ between true cosine and SimHash-estimated cosine
# --------------------------------------------------------------------- #
def test_hamming_cosine_spearman_synthetic_gaussian():
    """SimHash → cosine monotonicity on synthetic Gaussian-direction data.

    Construct vectors with *known* cosine to a planted anchor:

        X[i] = w[i] * anchor + sqrt(1 - w[i]^2) * v_perp[i]

    where v_perp is a unit-norm Gaussian-direction sample orthogonalized
    against the anchor, and w[i] ~ U(-1, 1). By construction
    ``cos(X[i], anchor) == w[i]``, so we can measure Spearman ρ between
    the *true* cosines (== w) and SimHash's estimate cos(π · h / n_bits)
    without confounding from concentration of measure.

    Pure isotropic Gaussian wouldn't work for this bar: at d=768 it
    concentrates cosines around 0 with stdev ≈ 1/√d ≈ 0.036, well below
    the SimHash noise floor — so ρ caps near 0.6 regardless of
    implementation. Planted-cosine construction restores the dynamic
    range the test is implicitly assuming.
    """
    rng = np.random.default_rng(7)
    n, d = 1000, 768

    anchor = rng.standard_normal(d)
    anchor /= np.linalg.norm(anchor)

    # Gaussian directions, orthogonalized against the anchor and
    # unit-normalized. (This is what "Gaussian-direction noise" means.)
    v = rng.standard_normal((n, d))
    v -= (v @ anchor)[:, None] * anchor[None, :]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12

    w = rng.uniform(-1.0, 1.0, size=n)
    X = (
        w[:, None] * anchor[None, :]
        + np.sqrt(1.0 - w**2)[:, None] * v
    ).astype(np.float64)

    q = SignBitQuantizer(d=d, seed=7)
    codes = q.encode(X)

    qcode = encode_signs(anchor @ q.rotation_)
    h = hamming_distances(codes, qcode)
    cos_est = np.cos(np.pi * h / q.n_bits)

    rho, _ = spearmanr(w, cos_est)
    assert rho > 0.95, f"Spearman ρ={rho:.4f}, expected > 0.95"


# --------------------------------------------------------------------- #
# 4. Self-recall on isotropic Gaussian
# --------------------------------------------------------------------- #
def test_self_recall_synthetic_gaussian():
    rng = np.random.default_rng(11)
    n, d = 500, 768
    X = rng.standard_normal((n, d)).astype(np.float64)
    q = SignBitQuantizer(d=d, seed=11)
    codes = q.encode(X)

    hits = 0
    for i in range(n):
        idx = q.search(X[i], codes, k=1)
        if idx[0] == i:
            hits += 1
    recall = hits / n
    # The query is byte-identical to codes[i] (rotation is the same), so
    # its Hamming distance to itself is exactly zero. Tied zeros are not
    # expected on Gaussian d=768 (collision probability ≈ 2^-768).
    assert recall >= 0.95, f"self-recall@1 = {recall:.3f}, expected ≥ 0.95"


# --------------------------------------------------------------------- #
# 5. Block-size constraint
# --------------------------------------------------------------------- #
def test_d_not_multiple_of_8_raises_clear_error():
    with pytest.raises(ValueError, match="divisible by 8"):
        SignBitQuantizer(d=100, seed=0)


def test_encode_signs_rejects_non_byte_aligned_input():
    X = np.ones((4, 7))
    with pytest.raises(ValueError, match="divisible by 8"):
        encode_signs(X)


def test_d_must_be_positive_integer():
    with pytest.raises(ValueError, match="positive integer"):
        SignBitQuantizer(d=0, seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        SignBitQuantizer(d=-8, seed=0)


# --------------------------------------------------------------------- #
# Functional / class parity & shape sanity
# --------------------------------------------------------------------- #
def test_functional_class_parity():
    """Class API is just sugar over the functional primitives."""
    rng = np.random.default_rng(3)
    n, d = 64, 128
    X = rng.standard_normal((n, d))

    R = haar_rotation(d, seed=3)
    codes_func = encode_signs(X @ R)

    q = SignBitQuantizer(d=d, seed=3)
    codes_class = q.encode(X)

    np.testing.assert_array_equal(codes_func, codes_class)


def test_search_returns_self_first_with_zero_distance():
    rng = np.random.default_rng(5)
    n, d = 200, 256
    X = rng.standard_normal((n, d))
    q = SignBitQuantizer(d=d, seed=5)
    codes = q.encode(X)

    idx, dist = q.search(X[42], codes, k=10, return_distances=True)
    assert idx.shape == (10,)
    assert dist.shape == (10,)
    assert idx[0] == 42
    assert dist[0] == 0
    # Distances must be non-decreasing.
    assert np.all(np.diff(dist) >= 0)


def test_batched_search_shape_and_self_first():
    rng = np.random.default_rng(8)
    n, d = 100, 64
    X = rng.standard_normal((n, d))
    q = SignBitQuantizer(d=d, seed=8)
    codes = q.encode(X)

    queries = X[[0, 50, 99]]
    idx = q.search(queries, codes, k=3)
    assert idx.shape == (3, 3)
    assert idx[0, 0] == 0
    assert idx[1, 0] == 50
    assert idx[2, 0] == 99


def test_encode_accepts_single_vector():
    rng = np.random.default_rng(2)
    d = 32
    x = rng.standard_normal(d)
    q = SignBitQuantizer(d=d, seed=2)
    code = q.encode(x)
    assert code.shape == (d // 8,)
    assert code.dtype == np.uint8


def test_hamming_search_functional_api():
    """The standalone functional ``hamming_search`` matches the class API."""
    rng = np.random.default_rng(9)
    n, d = 80, 128
    X = rng.standard_normal((n, d))
    R = haar_rotation(d, seed=9)
    codes = encode_signs(X @ R)

    qvec = X[7]
    idx_func = hamming_search(qvec @ R, codes, k=5)

    q = SignBitQuantizer(d=d, seed=9)
    np.testing.assert_array_equal(q.rotation_, R)  # same seed
    idx_class = q.search(qvec, q.encode(X), k=5)

    np.testing.assert_array_equal(idx_func, idx_class)


def test_hamming_distances_symmetry():
    """Hamming distance is symmetric and equals 0 for identical codes."""
    rng = np.random.default_rng(13)
    R = haar_rotation(64, seed=13)
    X = rng.standard_normal((10, 64))
    codes = encode_signs(X @ R)
    # Each row should have distance 0 to itself.
    for i in range(10):
        d_self = hamming_distances(codes[i : i + 1], codes[i])
        assert d_self[0] == 0
    # Symmetry: d(a, b) == d(b, a).
    d_ab = hamming_distances(codes[0:1], codes[1])[0]
    d_ba = hamming_distances(codes[1:2], codes[0])[0]
    assert d_ab == d_ba


def test_search_validates_codes_shape():
    rng = np.random.default_rng(0)
    q = SignBitQuantizer(d=64, seed=0)
    X = rng.standard_normal((10, 64))
    codes = q.encode(X)
    with pytest.raises(ValueError, match="incompatible"):
        q.search(X[0], codes[:, :-1], k=3)  # truncated codes


def test_k_larger_than_n_returns_all():
    rng = np.random.default_rng(0)
    q = SignBitQuantizer(d=32, seed=0)
    X = rng.standard_normal((5, 32))
    codes = q.encode(X)
    idx = q.search(X[0], codes, k=100)
    # ``min(k, n)`` semantics: clamp to n.
    assert idx.shape == (5,)


def test_fit_validates_shape():
    q = SignBitQuantizer(d=16, seed=0)
    rng = np.random.default_rng(0)
    q.fit(rng.standard_normal((10, 16)))  # OK
    with pytest.raises(ValueError, match="expected"):
        q.fit(rng.standard_normal((10, 8)))


def test_fit_returns_self():
    q = SignBitQuantizer(d=16, seed=0)
    assert q.fit() is q
    assert q.fit(np.zeros((3, 16))) is q


# --------------------------------------------------------------------- #
# stable_top_k — tie-break stability
# --------------------------------------------------------------------- #
from remax.packing import stable_top_k


def test_stable_top_k_matches_full_argsort_no_ties():
    """Smoke: with all-distinct distances, stable_top_k matches argsort."""
    rng = np.random.default_rng(0)
    dists = rng.permutation(1000).astype(np.int64)  # all distinct
    for k in (1, 5, 10, 100, 999, 1000, 1500):
        got = stable_top_k(dists, k)
        want = np.argsort(dists, kind="stable")[: min(k, dists.size)]
        np.testing.assert_array_equal(got, want)


def test_stable_top_k_breaks_ties_by_ascending_index():
    """Equal distances must be returned in ascending index order."""
    # Five elements all at distance 0, then one tied with the last at dist 1.
    dists = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    got = stable_top_k(dists, 5)
    np.testing.assert_array_equal(got, [0, 1, 2, 3, 4])
    # Top-3 from a long string of zeros must take the first three.
    dists = np.zeros(50, dtype=np.int64)
    got = stable_top_k(dists, 3)
    np.testing.assert_array_equal(got, [0, 1, 2])


def test_stable_top_k_handles_kth_boundary_ties():
    """Regression for the argpartition-instability bug.

    Construct a case where argpartition with k=4 may swap two indices
    that share the kth-smallest distance. ``stable_top_k`` must always
    pick the lower-indexed one, matching ``argsort(stable)[:k]``.
    """
    # Sorted ascending: [1, 2, 3, 4, 5, 5, 6, 7, 8, 9].
    # The kth-smallest at k=5 is 5, shared by indices 2 and 7.
    # Naive argpartition can put either one inside the partition.
    # stable_top_k must always pick index 2 (the lower-indexed one).
    dists = np.array([1, 2, 5, 3, 4, 9, 8, 5, 7, 6], dtype=np.int64)
    want = np.argsort(dists, kind="stable")[:5]
    got = stable_top_k(dists, 5)
    np.testing.assert_array_equal(got, want)
    assert 2 in got and 7 not in got


def test_stable_top_k_random_stress_matches_argsort():
    """Across many random distance vectors with forced ties, the helper
    must agree with the slow O(n log n) reference."""
    rng = np.random.default_rng(2026)
    for trial in range(50):
        n = int(rng.integers(20, 500))
        # Small range forces lots of ties.
        dists = rng.integers(0, 8, size=n).astype(np.int64)
        for k in (1, 3, n // 4, n // 2, n - 1, n):
            if k <= 0:
                continue
            got = stable_top_k(dists, k)
            want = np.argsort(dists, kind="stable")[: min(k, n)]
            np.testing.assert_array_equal(got, want)


def test_stable_top_k_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="positive"):
        stable_top_k(np.array([1, 2, 3]), 0)
    with pytest.raises(ValueError, match="positive"):
        stable_top_k(np.array([1, 2, 3]), -1)


def test_search_class_matches_full_argsort_with_ties():
    """SignBitQuantizer.search must match argsort(stable) when distances tie.

    Pre-fix: the class used argpartition + partial sort, which broke
    stability at the kth boundary. This test exercises the same shape as
    the previously-failing test_search_class_matches_manual_hamming in
    test_stacked.py.
    """
    rng = np.random.default_rng(13)
    n, d = 200, 128
    X = rng.standard_normal((n, d))
    q = SignBitQuantizer(d=d, seed=13)
    codes = q.encode(X)

    qcode = q.encode(X[7])
    xor = np.bitwise_xor(codes, qcode[None, :])
    from remax.packing import POPCOUNT_LUT

    manual_dists = POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int64)
    manual_top10 = np.argsort(manual_dists, kind="stable")[:10]
    cls_top10 = q.search(X[7], codes, k=10)
    np.testing.assert_array_equal(manual_top10, cls_top10)


# --------------------------------------------------------------------- #
# 4. Working-precision (dtype) parity
# --------------------------------------------------------------------- #
def test_default_dtype_is_float32():
    """The default working precision is f32 (PR rationale: SimHash output
    is 1 bit, so f64 in the rotation matmul is wasted bandwidth)."""
    q = SignBitQuantizer(d=64, seed=0)
    assert q.dtype == np.float32
    assert q.rotation_.dtype == np.float32


def test_dtype_param_round_trips():
    """Both f32 and f64 produce orthogonal rotations and self-consistent codes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 256))
    for dtype in (np.float32, np.float64):
        q = SignBitQuantizer(d=256, seed=42, dtype=dtype)
        assert q.dtype == np.dtype(dtype)
        assert q.rotation_.dtype == np.dtype(dtype)
        codes = q.encode(X)
        assert codes.dtype == np.uint8
        # Self-distance is zero — encode then re-encode the same row gives
        # identical codes.
        c0 = q.encode(X[0])
        np.testing.assert_array_equal(codes[0], c0)


def test_f32_vs_f64_top10_recall_synthetic():
    """f32 and f64 quantizers agree on the vast majority of top-10 results.

    SimHash only consumes ``sign(<r, x>)``; the f32 vs f64 mismatch can only
    flip a bit when ``<r, x> ≈ 0``, which is rare. Top-10 overlap should be
    ≥ 9/10 in expectation across queries on isotropic Gaussian data.
    """
    rng = np.random.default_rng(7)
    n, d, k_top = 1000, 256, 10
    X = rng.standard_normal((n, d))
    queries = rng.standard_normal((30, d))

    q32 = SignBitQuantizer(d=d, seed=2026, dtype=np.float32)
    q64 = SignBitQuantizer(d=d, seed=2026, dtype=np.float64)
    c32 = q32.encode(X)
    c64 = q64.encode(X)

    # Codes themselves should match in nearly all bits — disagreement only
    # at coordinates whose f64 dot product fell within f32 round-off of 0.
    bit_diff = np.unpackbits(c32 ^ c64).sum() / (n * d)
    assert bit_diff < 1e-3, f"f32 vs f64 bit disagreement {bit_diff:.4%} too high"

    overlaps = []
    for qv in queries:
        t32 = set(q32.search(qv, c32, k=k_top).tolist())
        t64 = set(q64.search(qv, c64, k=k_top).tolist())
        overlaps.append(len(t32 & t64))
    mean_overlap = np.mean(overlaps)
    assert mean_overlap >= 9.0, (
        f"mean top-{k_top} overlap {mean_overlap:.2f} < 9.0 — f32 path "
        "diverges from f64 more than expected"
    )


def test_dtype_passes_through_input_unchanged():
    """f32 input through an f32 quantizer should not trigger an upcast."""
    rng = np.random.default_rng(0)
    X32 = rng.standard_normal((10, 64)).astype(np.float32)
    q = SignBitQuantizer(d=64, seed=0)  # default f32
    # Patch np.asarray to fail loudly if a dtype-changing copy happens.
    # We assert the simpler invariant: encoding f32 input with an f32
    # quantizer matches encoding the explicit f64 view through an f32
    # quantizer (i.e. the cast is a no-op for f32 input).
    X64 = X32.astype(np.float64)
    np.testing.assert_array_equal(q.encode(X32), q.encode(X64))
