"""Tests for ``remax.sparse.SparseSignBitQuantizer`` (issue #33).

Required by the issue's "RED tests" section:

  * Shape on random CSR for k ∈ {64, 128, 256}.
  * Determinism: same (seed, X) → byte-equal output.
  * Seed sensitivity: different seeds → different output.
  * ``k % 8 != 0`` raises in ``__init__``.
  * Empty row encodes deterministically (all-zero bytes pre-centering).
  * Hand-derived bit-pattern test: 3 docs × 5 terms × k=8, seed=0 —
    expected bytes computed from the SplitMix64 spec, not from the
    encoder.
  * JL sanity: rows sharing 90% nonzeros have lower Hamming distance
    than rows sharing 0%, across 5 seeds at k=256.

Specification-gap note
----------------------
The hand-derived test re-implements SplitMix64 *in the test file* and
derives bucket / sign / accumulated buffer / packed bytes step by step
from the spec. The encoder is a black box to that derivation — if the
encoder uses a different hash, the test fails. This is the discipline
called out by issue #33: "If you find yourself running the encoder to
populate expected values, the test is bolt-on — stop and re-derive."
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sp_random

from remax.sparse import SparseSignBitQuantizer


# --------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [64, 128, 256])
def test_encode_shape_random_csr(k: int):
    rng = np.random.default_rng(0)
    n, d = 50, 10_000
    X = sp_random(n, d, density=0.01, format="csr", random_state=rng)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    codes = enc.encode(X)
    assert codes.shape == (n, k // 8)
    assert codes.dtype == np.uint8


# --------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------- #
def test_determinism_byte_equal():
    rng = np.random.default_rng(7)
    X = sp_random(20, 500, density=0.02, format="csr", random_state=rng)
    e1 = SparseSignBitQuantizer(d=500, k=128, seed=42)
    e2 = SparseSignBitQuantizer(d=500, k=128, seed=42)
    c1 = e1.encode(X)
    c2 = e2.encode(X)
    assert c1.tobytes() == c2.tobytes()
    np.testing.assert_array_equal(c1, c2)


def test_seed_sensitivity():
    rng = np.random.default_rng(3)
    X = sp_random(20, 500, density=0.05, format="csr", random_state=rng)
    c0 = SparseSignBitQuantizer(d=500, k=128, seed=0).encode(X)
    c1 = SparseSignBitQuantizer(d=500, k=128, seed=1).encode(X)
    assert c0.tobytes() != c1.tobytes()


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
def test_k_not_multiple_of_8_raises():
    with pytest.raises(ValueError, match="multiple of 8"):
        SparseSignBitQuantizer(d=100, k=10, seed=0)


def test_d_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        SparseSignBitQuantizer(d=0, k=8, seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        SparseSignBitQuantizer(d=-1, k=8, seed=0)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        SparseSignBitQuantizer(d=10, k=0, seed=0)


def test_dense_input_rejected():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    with pytest.raises(TypeError, match="scipy.sparse"):
        enc.encode(np.zeros((3, 10)))


def test_shape_mismatch_rejected():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    X = csr_matrix(np.zeros((3, 9)))
    with pytest.raises(ValueError, match="expected"):
        enc.encode(X)


# --------------------------------------------------------------------- #
# Empty rows
# --------------------------------------------------------------------- #
def test_empty_row_encodes_to_zero_bytes_default():
    """A row with no nonzeros encodes to all-zero bytes (center=False)."""
    n, d, k = 4, 20, 16
    rng = np.random.default_rng(0)
    X = sp_random(n, d, density=0.3, format="csr", random_state=rng)
    # Force row 2 to be empty.
    X = X.tolil()
    X[2, :] = 0
    X = X.tocsr()
    X.eliminate_zeros()
    assert X.getrow(2).nnz == 0

    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    codes = enc.encode(X)
    assert np.all(codes[2] == 0)


def test_all_empty_matrix_encodes_to_zero_bytes_default():
    """An n × d sparse matrix with zero nonzeros encodes to all zeros."""
    n, d, k = 5, 32, 32
    X = csr_matrix((n, d))
    assert X.nnz == 0
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    codes = enc.encode(X)
    assert codes.shape == (n, k // 8)
    assert np.all(codes == 0)


# --------------------------------------------------------------------- #
# Hand-derived bit-pattern test
# --------------------------------------------------------------------- #
def test_hand_computed_pattern_3x5_k8_seed0():
    """Hand-derived expected bytes from the SplitMix64 spec.

    Spec (issue #33 "Mechanics"):
      bucket[j] = H1(seed, j) % k
      sign[j]   = ±1 from a second independent hash H2(seed, j)

    Concrete hash:
      salt_a = SplitMix64(seed)
      salt_b = SplitMix64(seed XOR 0x9E3779B97F4A7C15)
      H1(s, j) = SplitMix64((salt_a + j) mod 2^64)
      H2(s, j) = SplitMix64((salt_b + j) mod 2^64)
      sign[j]  = +1 if (H2 & 1) == 0 else -1

    This test re-implements SplitMix64 here, derives bucket / sign /
    buffer / bytes step by step, and asserts equality with the encoder
    output. The encoder is a black box to the derivation.
    """
    # ---- Reference SplitMix64 (Steele, Lea, Flood, JCSS 2014) ----
    MASK64 = (1 << 64) - 1
    C0 = 0x9E3779B97F4A7C15
    C1 = 0xBF58476D1CE4E3B9
    C2 = 0x94D049BB133111EB

    def smix(x: int) -> int:
        x = (x + C0) & MASK64
        x = ((x ^ (x >> 30)) * C1) & MASK64
        x = ((x ^ (x >> 27)) * C2) & MASK64
        return x ^ (x >> 31)

    seed, d, k = 0, 5, 8
    salt_a = smix(seed)
    salt_b = smix(seed ^ C0)
    expected_bucket = np.array(
        [smix((salt_a + j) & MASK64) % k for j in range(d)], dtype=np.int64
    )
    expected_sign = np.array(
        [+1 if (smix((salt_b + j) & MASK64) & 1) == 0 else -1
         for j in range(d)],
        dtype=np.int8,
    )

    # These values are determined entirely by the spec above. They are
    # written down here as a fixed checkpoint so a future reader can
    # verify them without re-running the test.
    assert expected_bucket.tolist() == [1, 0, 1, 5, 1]
    assert expected_sign.tolist() == [-1, -1, 1, 1, -1]

    # ---- Input docs ----
    docs = np.array(
        [
            [1.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ---- Hand-accumulate the length-k buffer per doc ----
    expected_buf = np.zeros((3, k), dtype=np.float64)
    for i in range(3):
        for j in range(d):
            expected_buf[i, expected_bucket[j]] += (
                expected_sign[j] * docs[i, j]
            )

    # Buffers (for the reader): each row has length 8.
    #   doc0: [0, 1, 0, 0, 0, 0, 0, 0]            (sign[2]*2 + sign[0]*1 → buf[1])
    #   doc1: [-3, 0, 0, 0, 0, 4, 0, 0]            (sign[1]*3 → buf[0]; sign[3]*4 → buf[5])
    #   doc2: [-1, -1, 0, 0, 0, 1, 0, 0]           (sum of all five terms)
    np.testing.assert_array_equal(
        expected_buf[0], [0, 1, 0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        expected_buf[1], [-3, 0, 0, 0, 0, 4, 0, 0]
    )
    np.testing.assert_array_equal(
        expected_buf[2], [-1, -1, 0, 0, 0, 1, 0, 0]
    )

    # ---- Pack: bit_b = (buf[b] > 0), big-endian within byte ----
    expected_bits = expected_buf > 0
    expected_bytes = np.packbits(expected_bits, axis=-1)
    # doc0 bits = 01000000 = 0x40
    # doc1 bits = 00000100 = 0x04
    # doc2 bits = 00000100 = 0x04
    np.testing.assert_array_equal(
        expected_bytes, np.array([[0x40], [0x04], [0x04]], dtype=np.uint8)
    )

    # ---- Encoder output ----
    enc = SparseSignBitQuantizer(d=d, k=k, seed=seed)
    np.testing.assert_array_equal(enc.bucket_, expected_bucket)
    np.testing.assert_array_equal(enc.sign_, expected_sign)
    got = enc.encode(csr_matrix(docs))
    np.testing.assert_array_equal(got, expected_bytes)


# --------------------------------------------------------------------- #
# JL sanity: similar rows have lower Hamming than dissimilar rows
# --------------------------------------------------------------------- #
def test_jl_sanity_similar_lower_hamming_than_dissimilar():
    """Hamming(0,1) < Hamming(0,2) across 5 seeds at k=256.

    Rows 0 and 1 share 90% of nonzeros (with identical positive values);
    rows 0 and 2 share 0% nonzeros. At k=256 the expected Hamming
    distance for the high-overlap pair is ≈ k * θ/π with cos ≈ 0.9
    (≈ 36 bits), versus ≈ 128 for the orthogonal pair. The margin is so
    wide that every reasonable seed should rank them correctly.
    """
    from remax import hamming_distances

    rng = np.random.default_rng(2026)
    d = 5_000
    k = 256

    nz_common = rng.choice(d, size=90, replace=False)
    remaining = np.setdiff1d(np.arange(d), nz_common)
    nz_only_1 = rng.choice(remaining, size=10, replace=False)
    remaining = np.setdiff1d(remaining, nz_only_1)
    nz_only_2 = rng.choice(remaining, size=100, replace=False)

    n, dim = 3, d
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    # row 0: 100 nonzeros (90 common + 10 unique to row 0)
    nz_only_0 = rng.choice(
        np.setdiff1d(remaining, nz_only_2), size=10, replace=False
    )
    for j in np.concatenate([nz_common, nz_only_0]):
        rows.append(0); cols.append(int(j)); data.append(1.0)
    # row 1: shares 90 with row 0, plus 10 unique to row 1 → 90 / 100 overlap
    for j in np.concatenate([nz_common, nz_only_1]):
        rows.append(1); cols.append(int(j)); data.append(1.0)
    # row 2: shares 0 with row 0 (entirely disjoint)
    for j in nz_only_2:
        rows.append(2); cols.append(int(j)); data.append(1.0)

    X = csr_matrix((data, (rows, cols)), shape=(n, dim), dtype=np.float64)

    for seed in range(5):
        enc = SparseSignBitQuantizer(d=dim, k=k, seed=seed)
        codes = enc.encode(X)
        d01 = int(hamming_distances(codes[0:1], codes[1])[0])
        d02 = int(hamming_distances(codes[0:1], codes[2])[0])
        assert d01 < d02, (
            f"seed={seed}: Hamming(0,1)={d01} not < Hamming(0,2)={d02}; "
            "JL property violated"
        )


# --------------------------------------------------------------------- #
# encode_query: convenience wrapper for single-row input
# --------------------------------------------------------------------- #
def test_encode_query_matches_encode_first_row():
    rng = np.random.default_rng(0)
    X = sp_random(5, 200, density=0.1, format="csr", random_state=rng)
    enc = SparseSignBitQuantizer(d=200, k=64, seed=0)
    codes = enc.encode(X)
    q = X.getrow(2)
    qcode = enc.encode_query(q)
    assert qcode.shape == (64 // 8,)
    np.testing.assert_array_equal(qcode, codes[2])


def test_encode_query_rejects_multi_row():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    X = csr_matrix(np.eye(3, 10))
    with pytest.raises(ValueError, match="single-row"):
        enc.encode_query(X)


# --------------------------------------------------------------------- #
# fit: only meaningful with center=True
# --------------------------------------------------------------------- #
def test_fit_returns_self():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    X = csr_matrix(np.eye(3, 10))
    assert enc.fit(X) is enc


def test_fit_no_op_for_center_false():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0, center=False)
    X = csr_matrix(np.eye(3, 10))
    enc.fit(X)
    assert enc.mean_buf_ is None


def test_center_true_requires_fit_before_encode():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0, center=True)
    X = csr_matrix(np.eye(3, 10))
    with pytest.raises(RuntimeError, match="fit"):
        enc.encode(X)


def test_center_true_subtracts_projected_mean():
    """Centered encode equals raw projection minus projected-mean vector.

    A row equal to the corpus mean encodes to the same buffer (≈ 0) as
    a true zero row would under center=False. Verify by feeding a
    one-row matrix whose row IS the mean: its buffer minus the projected
    mean should be exactly zero → all bits False → all-zero bytes.
    """
    rng = np.random.default_rng(0)
    n, d, k = 20, 100, 16
    X = sp_random(n, d, density=0.2, format="csr", random_state=rng)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0, center=True).fit(X)
    assert enc.mean_buf_ is not None
    assert enc.mean_buf_.shape == (k,)

    # Feed the mean row itself — accumulated buf equals mean_buf_,
    # so post-centering buf is zero, packbits → zero bytes.
    mean_row = csr_matrix(
        np.asarray(X.mean(axis=0)).ravel()[None, :]
    )
    code = enc.encode_query(mean_row)
    np.testing.assert_array_equal(code, np.zeros(k // 8, dtype=np.uint8))


# --------------------------------------------------------------------- #
# Compatibility with remax.hamming_distances
# --------------------------------------------------------------------- #
def test_codes_compatible_with_hamming_distances():
    """The encoded codes are a drop-in for the existing Hamming path."""
    from remax import hamming_distances

    rng = np.random.default_rng(0)
    X = sp_random(10, 200, density=0.1, format="csr", random_state=rng)
    enc = SparseSignBitQuantizer(d=200, k=64, seed=0)
    codes = enc.encode(X)
    dists = hamming_distances(codes, codes[0])
    assert dists.shape == (10,)
    assert dists[0] == 0  # self-distance is zero
    assert dists.dtype == np.int64
