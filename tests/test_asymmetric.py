"""Tests for asymmetric (float-query vs sign-bit-corpus) scoring.

The corpus side is unchanged -- still one bit per dimension. Only the query
stays in float. See ``remax.packing.asymmetric_scores`` for why that is free,
and ``bench/asymmetric_lfm25.py`` for the measured retrieval effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax import SignBitQuantizer, StackedSignBitQuantizer, asymmetric_scores
from remax.packing import asymmetric_search, encode_signs


def _dense_reference(q_rot, X_rot):
    """The thing the LUT is a fast path for: q . sign(X), densely."""
    return q_rot @ np.where(X_rot > 0, 1.0, -1.0).T


class TestAsymmetricScores:
    def test_matches_dense_reference(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 128)).astype(np.float32)
        q = rng.standard_normal(128).astype(np.float32)
        got = asymmetric_scores(q, encode_signs(X))
        np.testing.assert_allclose(got, _dense_reference(q, X), atol=1e-3)

    def test_chunking_does_not_change_result(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((997, 64)).astype(np.float32)
        q = rng.standard_normal(64).astype(np.float32)
        codes = encode_signs(X)
        np.testing.assert_allclose(
            asymmetric_scores(q, codes, chunk=7),
            asymmetric_scores(q, codes, chunk=1 << 16),
            atol=1e-4,
        )

    def test_scores_are_true_inner_products_not_just_rank_correct(self):
        # q . s == 2*(q . b) - sum(q); the affine term is applied, so callers
        # that threshold or fuse these get a real inner product.
        rng = np.random.default_rng(2)
        X = rng.standard_normal((64, 32)).astype(np.float32)
        q = rng.standard_normal(32).astype(np.float32)
        got = asymmetric_scores(q, encode_signs(X))
        assert got.min() >= -np.abs(q).sum() - 1e-3
        assert got.max() <= np.abs(q).sum() + 1e-3
        np.testing.assert_allclose(got, _dense_reference(q, X), atol=1e-3)

    def test_rejects_dimension_mismatch(self):
        codes = encode_signs(np.zeros((4, 32), dtype=np.float32))
        with pytest.raises(ValueError, match="bits"):
            asymmetric_scores(np.zeros(64, dtype=np.float32), codes)

    def test_rejects_non_uint8_codes(self):
        with pytest.raises(ValueError, match="uint8"):
            asymmetric_scores(np.zeros(32, np.float32),
                              np.zeros((4, 4), dtype=np.int32))


class TestAsymmetricSearch:
    def test_finds_self_first(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(X)
        for i in (0, 17, 299):
            assert q.search_asymmetric(X[i], codes, k=5)[0] == i

    def test_descending_by_score(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(X)
        _, scores = q.search_asymmetric(X[0], codes, k=10, return_scores=True)
        assert np.all(np.diff(scores) <= 1e-5)

    def test_batch_shape_and_squeeze(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((120, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(X)
        assert q.search_asymmetric(X[0], codes, k=4).shape == (4,)
        assert q.search_asymmetric(X[:6], codes, k=4).shape == (6, 4)

    def test_reads_the_same_index_as_hamming_search(self):
        # The whole point: no re-encoding, no second index.
        rng = np.random.default_rng(6)
        X = rng.standard_normal((200, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(X)
        assert q.search(X[3], codes, k=5)[0] == 3
        assert q.search_asymmetric(X[3], codes, k=5)[0] == 3

    def test_functional_api_agrees_with_method(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((150, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(X)
        np.testing.assert_array_equal(
            q.search_asymmetric(X[0], codes, k=5),
            asymmetric_search(X[0] @ q.rotation_, codes, k=5),
        )

    def test_k_larger_than_corpus_is_clamped(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((5, 64)).astype(np.float32)
        q = SignBitQuantizer(d=64, seed=42)
        assert q.search_asymmetric(X[0], q.encode(X), k=50).shape == (5,)

    def test_rejects_bad_k(self):
        q = SignBitQuantizer(d=64, seed=42)
        codes = q.encode(np.zeros((4, 64), dtype=np.float32))
        with pytest.raises(ValueError, match="k must be positive"):
            q.search_asymmetric(np.zeros(64, np.float32), codes, k=0)


class TestAsymmetricRecall:
    def test_beats_symmetric_on_isotropic_gaussian(self):
        """The claim that justifies the method existing.

        Asymmetric keeps the query in float, so it should recover more of the
        exact-cosine top-k than Hamming does at the identical stored index.
        Measured against the fp32 ranking on isotropic data at a deliberately
        lossy width -- the regime where the bench shows the gap is widest.
        """
        rng = np.random.default_rng(9)
        n, d, k = 2000, 128, 10
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = rng.standard_normal((100, d)).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)

        truth = np.argsort(-(Q @ X.T), axis=1)[:, :k]
        quant = SignBitQuantizer(d=d, seed=42)
        codes = quant.encode(X)

        sym = quant.search(Q, codes, k=k)
        asym = quant.search_asymmetric(Q, codes, k=k)
        r_sym = np.mean([len(set(a) & set(b)) / k for a, b in zip(sym, truth)])
        r_asym = np.mean([len(set(a) & set(b)) / k for a, b in zip(asym, truth)])
        assert r_asym > r_sym, f"asym {r_asym:.3f} !> sym {r_sym:.3f}"


class TestStackedAsymmetric:
    def test_finds_self_first(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 64)).astype(np.float32)
        q = StackedSignBitQuantizer(d=64, k=4, seed=42)
        codes = q.encode(X)
        assert q.search_asymmetric(X[0], codes, k=5)[0] == 0

    def test_matches_dense_reference_over_stacked_rotations(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((150, 64)).astype(np.float32)
        q = StackedSignBitQuantizer(d=64, k=2, seed=42)
        codes = q.encode(X)
        got = asymmetric_scores(X[0] @ q._rotation_matrix, codes)
        ref = _dense_reference(X[0] @ q._rotation_matrix, X @ q._rotation_matrix)
        np.testing.assert_allclose(got, ref, atol=1e-2)

    def test_rejects_wrong_width_codes(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((20, 64)).astype(np.float32)
        flat = SignBitQuantizer(d=64, seed=42).encode(X)  # k=1 width
        q = StackedSignBitQuantizer(d=64, k=4, seed=42)
        with pytest.raises(ValueError, match="incompatible"):
            q.search_asymmetric(X[0], flat, k=3)
