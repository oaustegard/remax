"""Tests for ``remax.stacked.StackedSignBitQuantizer`` (issue #3).

Required by the issue's "Tests required" section:

  1. Independence: pairwise different rotations under k=4.
  2. Variance reduction on synthetic: var of Hamming-derived cosine
     estimator scales ~1/k for k ∈ {1, 2, 4, 8}.
  3. Recall monotonicity on synthetic Gaussian: R@10 increases
     monotonically in k for k ∈ {1, 2, 4, 8}.
  4. Encoded size: ``encode(X).shape == (n, k * d // 8)``.

Plus parity tests for: input validation, seed determinism, sklearn-style
ergonomics, single-vector and batched search.

Note on synthetic data
----------------------
The same low-rank Gaussian construction used in the 1-bit smoke benchmark
applies here. Pure isotropic Gaussian at d=768 concentrates pairwise
cosines around zero with stdev ≈ 1/√d, which is below the SimHash noise
floor — recall@10 caps near chance regardless of stack count, so it
tells us nothing about k-monotonicity.

The recall-monotonicity test (#3) uses d=256, n=2000, subdim=16 to keep
pytest fast (~0.5s). The full d=768, n=10k baseline from the issue lives
in ``bench/smoke_stacked.py``; documented baselines below come from that
script (seed=42).
"""

from __future__ import annotations

import numpy as np
import pytest

from remax import StackedSignBitQuantizer
from remax.packing import POPCOUNT_LUT


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _low_rank_gaussian(n: int, d: int, subdim: int, *, rng: np.random.Generator,
                       basis: np.ndarray | None = None):
    """Draw n points from a subdim-dim Gaussian embedded in R^d.

    Same construction as bench/smoke_1bit.py: gives a continuous,
    non-degenerate spread of pairwise cosines without the
    concentration-of-measure issues of a full-rank Gaussian at d=768.

    If ``basis`` is provided, draws are made in that fixed subspace —
    needed when corpus and queries must share structure for retrieval to
    be meaningful.
    """
    if basis is None:
        basis, _ = np.linalg.qr(rng.standard_normal((d, subdim)))
    return (rng.standard_normal((n, subdim)) @ basis.T).astype(np.float64), basis


def _recall_at_k(true_topk: np.ndarray, pred_topk: np.ndarray) -> float:
    k = true_topk.shape[1]
    total = sum(
        np.intersect1d(t, p, assume_unique=False).size
        for t, p in zip(true_topk, pred_topk)
    )
    return total / (true_topk.shape[0] * k)


# --------------------------------------------------------------------- #
# 1. Independence of stacked rotations
# --------------------------------------------------------------------- #
def test_pairwise_independent_rotations():
    """Each of the k Haar rotations differs from the others.

    Independence here is the practical kind: the SeedSequence-derived
    child seeds drive distinct RNG streams, so the resulting Haar matrices
    are different to within their full Frobenius range. Pairwise frobenius
    distance > 0 (in fact ≫ 0) is the operative check.
    """
    q = StackedSignBitQuantizer(d=128, k=4, seed=0)
    assert q.rotations_.shape == (4, 128, 128)
    for i in range(4):
        for j in range(i + 1, 4):
            diff = np.linalg.norm(q.rotations_[i] - q.rotations_[j])
            assert diff > 1e-3, (
                f"rotations {i} and {j} are suspiciously similar (frob diff={diff})"
            )


def test_each_stacked_rotation_is_orthogonal():
    """Every per-stack Haar rotation must be exactly orthogonal."""
    q = StackedSignBitQuantizer(d=64, k=8, seed=7)
    eye = np.eye(64)
    atol = 8 * np.sqrt(64) * np.finfo(q.rotations_.dtype).eps
    for j in range(8):
        np.testing.assert_allclose(
            q.rotations_[j] @ q.rotations_[j].T, eye, atol=atol
        )


# --------------------------------------------------------------------- #
# 2. Variance reduction with k
# --------------------------------------------------------------------- #
def test_variance_reduction_synthetic():
    """Sample variance of the Hamming-derived disagreement-rate estimator
    should scale ~1/k as k grows.

    Construction
    ------------
    Fix two unit vectors a, b with cos(a, b) = 0.6 → θ = arccos(0.6).
    For each (k, master_seed) pair, encode a and b under
    StackedSignBitQuantizer(d, k, seed=master_seed) and compute
    p̂ = hamming_distance / (k · d) — an estimator of θ/π.

    Theory: Var(p̂) = (θ/π)(1 − θ/π) / (k · d), so Var(p̂) ∝ 1/k for fixed d.

    Empirically (M=200 trials, d=128, see baseline run in
    bench/smoke_stacked.py):
        k=1  var ≈ 1.10e-3
        k=2  var ≈ 5.6e-4
        k=4  var ≈ 2.7e-4
        k=8  var ≈ 1.3e-4
    Ratio var(k=1)/var(k=8) ≈ 8.4 (theory: 8). The test asserts > 4, with
    slack for the M=200 sample-variance estimator's own variance.
    """
    d = 128
    rng = np.random.default_rng(0)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)

    target_cos = 0.6
    v = rng.standard_normal(d)
    v -= (v @ a) * a
    v /= np.linalg.norm(v)
    b = target_cos * a + np.sqrt(1.0 - target_cos**2) * v
    np.testing.assert_allclose(a @ b, target_cos, atol=1e-12)

    theta = np.arccos(target_cos)
    true_p = theta / np.pi

    M = 200
    variances: dict[int, float] = {}
    means: dict[int, float] = {}
    for k_stacks in (1, 2, 4, 8):
        estimates = np.empty(M)
        for trial in range(M):
            q = StackedSignBitQuantizer(d=d, k=k_stacks, seed=trial)
            ca = q.encode(a)
            cb = q.encode(b)
            h = np.unpackbits(np.bitwise_xor(ca, cb)).sum()
            estimates[trial] = h / q.n_bits
        variances[k_stacks] = float(estimates.var())
        means[k_stacks] = float(estimates.mean())

    # All k means concentrate around true θ/π — within 0.02 at M=200.
    for k_stacks, m in means.items():
        assert abs(m - true_p) < 0.02, (
            f"k={k_stacks}: empirical mean {m:.4f} far from θ/π {true_p:.4f}"
        )

    # Monotonic decrease: each step should at least halve variance modulo
    # sample noise. Use 0.65 as a generous bound (theory: 0.5).
    for prev_k, next_k in [(1, 2), (2, 4), (4, 8)]:
        ratio = variances[next_k] / variances[prev_k]
        assert ratio < 0.65, (
            f"variance did not shrink enough from k={prev_k} to k={next_k}: "
            f"var ratio {ratio:.3f} (expected < 0.65, theory ≈ 0.5)"
        )

    # End-to-end: var(k=1)/var(k=8) should be ~8 by theory; assert > 4.
    overall_ratio = variances[1] / variances[8]
    assert overall_ratio > 4.0, (
        f"var(k=1)/var(k=8) = {overall_ratio:.2f}, expected > 4 (theory: 8)"
    )


# --------------------------------------------------------------------- #
# 3. Recall monotonicity in k
# --------------------------------------------------------------------- #
def test_recall_monotonicity_synthetic_gaussian():
    """R@10 on low-rank Gaussian must increase monotonically in k.

    Configuration: d=256, n=2000, subdim=16, queries=100, master seed=11.
    Baselines from this exact config (seed=11):
        k=1  R@10 ≈ 0.67
        k=2  R@10 ≈ 0.74
        k=4  R@10 ≈ 0.82
        k=8  R@10 ≈ 0.87

    Issue #3 spec mentions d=768, n=10k as the canonical configuration.
    That run is preserved as a smoke benchmark in bench/smoke_stacked.py;
    the unit test uses a smaller config so pytest stays fast (~0.5s).
    """
    rng = np.random.default_rng(11)
    n, d, subdim, queries = 2000, 256, 16, 100
    X, basis = _low_rank_gaussian(n, d, subdim, rng=rng)
    Q, _ = _low_rank_gaussian(queries, d, subdim, rng=rng, basis=basis)

    # Float ground truth via cosine similarity.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    sims = Qn @ Xn.T
    true_topk = np.argpartition(-sims, 10, axis=1)[:, :10]

    recalls: dict[int, float] = {}
    for k_stacks in (1, 2, 4, 8):
        q = StackedSignBitQuantizer(d=d, k=k_stacks, seed=11)
        codes = q.encode(X)
        # Sanity: encoded width matches the formula.
        assert codes.shape == (n, k_stacks * (d // 8))
        pred_topk = q.search(Q, codes, k=10)
        recalls[k_stacks] = _recall_at_k(true_topk, pred_topk)

    # Strict monotonicity.
    seq = [recalls[k] for k in (1, 2, 4, 8)]
    for prev_, next_ in zip(seq, seq[1:]):
        assert next_ > prev_, (
            f"recall not monotone: {seq} (k=1,2,4,8)"
        )

    # And the magnitudes should be in the right ballpark — guards against a
    # silent regression that keeps monotonicity but tanks absolute recall.
    assert recalls[1] > 0.55, f"R@10 at k=1 unexpectedly low: {recalls[1]:.3f}"
    assert recalls[8] > 0.80, f"R@10 at k=8 unexpectedly low: {recalls[8]:.3f}"


# --------------------------------------------------------------------- #
# 4. Encoded size
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("d,k", [(8, 1), (64, 4), (768, 8), (256, 16)])
def test_encoded_size_matches_formula(d: int, k: int):
    rng = np.random.default_rng(0)
    n = 23
    X = rng.standard_normal((n, d))
    q = StackedSignBitQuantizer(d=d, k=k, seed=0)
    codes = q.encode(X)
    assert codes.shape == (n, k * (d // 8))
    assert codes.dtype == np.uint8


def test_encode_single_vector_returns_1d_code():
    rng = np.random.default_rng(0)
    d, k = 64, 3
    x = rng.standard_normal(d)
    q = StackedSignBitQuantizer(d=d, k=k, seed=0)
    code = q.encode(x)
    assert code.shape == (k * (d // 8),)
    assert code.dtype == np.uint8


# --------------------------------------------------------------------- #
# Determinism & validation
# --------------------------------------------------------------------- #
def test_seed_determinism_byte_identical():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 128))
    c1 = StackedSignBitQuantizer(d=128, k=4, seed=314).encode(X)
    c2 = StackedSignBitQuantizer(d=128, k=4, seed=314).encode(X)
    assert c1.tobytes() == c2.tobytes()
    np.testing.assert_array_equal(c1, c2)


def test_different_seeds_produce_different_codes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 128))
    c1 = StackedSignBitQuantizer(d=128, k=4, seed=1).encode(X)
    c2 = StackedSignBitQuantizer(d=128, k=4, seed=2).encode(X)
    assert c1.tobytes() != c2.tobytes()


def test_different_k_produces_different_widths():
    """k controls the per-row byte width; codes from different k cannot
    accidentally compare equal."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 64))
    c1 = StackedSignBitQuantizer(d=64, k=2, seed=0).encode(X)
    c2 = StackedSignBitQuantizer(d=64, k=4, seed=0).encode(X)
    assert c1.shape != c2.shape


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
def test_d_not_multiple_of_8_raises():
    with pytest.raises(ValueError, match="divisible by 8"):
        StackedSignBitQuantizer(d=100, k=2, seed=0)


def test_d_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        StackedSignBitQuantizer(d=0, k=2, seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        StackedSignBitQuantizer(d=-8, k=2, seed=0)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        StackedSignBitQuantizer(d=8, k=0, seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        StackedSignBitQuantizer(d=8, k=-1, seed=0)


def test_encode_validates_dim():
    q = StackedSignBitQuantizer(d=16, k=2, seed=0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="expected 16"):
        q.encode(rng.standard_normal((4, 8)))


def test_encode_rejects_3d_input():
    q = StackedSignBitQuantizer(d=8, k=2, seed=0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        q.encode(np.zeros((2, 3, 8)))


def test_search_validates_codes_shape():
    rng = np.random.default_rng(0)
    q = StackedSignBitQuantizer(d=64, k=3, seed=0)
    X = rng.standard_normal((10, 64))
    codes = q.encode(X)
    with pytest.raises(ValueError, match="incompatible"):
        # Drop a byte from each row → no longer matches k * d // 8.
        q.search(X[0], codes[:, :-1], k=3)


def test_search_k_must_be_positive():
    rng = np.random.default_rng(0)
    q = StackedSignBitQuantizer(d=8, k=2, seed=0)
    X = rng.standard_normal((4, 8))
    codes = q.encode(X)
    with pytest.raises(ValueError, match="k must be positive"):
        q.search(X[0], codes, k=0)


# --------------------------------------------------------------------- #
# sklearn-style ergonomics
# --------------------------------------------------------------------- #
def test_fit_returns_self_and_validates():
    q = StackedSignBitQuantizer(d=16, k=3, seed=0)
    assert q.fit() is q
    rng = np.random.default_rng(0)
    assert q.fit(rng.standard_normal((5, 16))) is q
    with pytest.raises(ValueError, match="expected"):
        q.fit(rng.standard_normal((5, 8)))


# --------------------------------------------------------------------- #
# Search behaviour
# --------------------------------------------------------------------- #
def test_search_self_first_with_zero_distance():
    rng = np.random.default_rng(5)
    n, d = 200, 128
    X = rng.standard_normal((n, d))
    q = StackedSignBitQuantizer(d=d, k=4, seed=5)
    codes = q.encode(X)
    idx, dist = q.search(X[42], codes, k=10, return_distances=True)
    assert idx.shape == (10,)
    assert dist.shape == (10,)
    assert idx[0] == 42
    assert dist[0] == 0
    # Distances must be non-decreasing.
    assert np.all(np.diff(dist) >= 0)
    # Distance ceiling: k * d bits.
    assert dist.max() <= q.n_bits


def test_batched_search_shape_and_self_first():
    rng = np.random.default_rng(8)
    n, d = 100, 64
    X = rng.standard_normal((n, d))
    q = StackedSignBitQuantizer(d=d, k=2, seed=8)
    codes = q.encode(X)

    queries = X[[0, 50, 99]]
    idx = q.search(queries, codes, k=3)
    assert idx.shape == (3, 3)
    assert idx[0, 0] == 0
    assert idx[1, 0] == 50
    assert idx[2, 0] == 99


def test_k_larger_than_n_returns_all():
    rng = np.random.default_rng(0)
    q = StackedSignBitQuantizer(d=32, k=2, seed=0)
    X = rng.standard_normal((5, 32))
    codes = q.encode(X)
    idx = q.search(X[0], codes, k=100)
    assert idx.shape == (5,)


# --------------------------------------------------------------------- #
# Encoding layout — concatenation invariant
# --------------------------------------------------------------------- #
def test_encoded_layout_is_per_row_concatenation():
    """The first d//8 bytes of each row must be the SimHash code under
    rotation 0; the next d//8 bytes under rotation 1; and so on. This is
    the public contract documented in the encode() docstring.
    """
    rng = np.random.default_rng(2)
    d, k, n = 64, 3, 7
    X = rng.standard_normal((n, d))
    q = StackedSignBitQuantizer(d=d, k=k, seed=2)
    codes = q.encode(X)

    bytes_per_stack = d // 8
    for j in range(k):
        # Compute the expected slice manually for stack j.
        rotated = X @ q.rotations_[j]  # (n, d)
        expected = np.packbits(rotated > 0, axis=-1)  # (n, d//8)
        actual = codes[:, j * bytes_per_stack : (j + 1) * bytes_per_stack]
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"stack {j} mismatch in encoded layout"
        )


def test_search_class_matches_manual_hamming():
    """Top-k from .search() agrees with manual Hamming over the full code."""
    rng = np.random.default_rng(13)
    n, d, k = 200, 128, 4
    X = rng.standard_normal((n, d))
    q = StackedSignBitQuantizer(d=d, k=k, seed=13)
    codes = q.encode(X)

    qvec = X[7]
    qcode = q.encode(qvec)
    # Manual Hamming distance using POPCOUNT_LUT — same primitive used inside.
    xor = np.bitwise_xor(codes, qcode[None, :])
    manual_dists = POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int64)
    manual_top10 = np.argsort(manual_dists, kind="stable")[:10]

    cls_top10 = q.search(qvec, codes, k=10)
    np.testing.assert_array_equal(manual_top10, cls_top10)
