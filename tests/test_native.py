"""Tests for remax._native — hardware-accelerated Hamming scan."""

import numpy as np
import pytest

from remax._native import AVAILABLE, hamming_distances_native
from remax.packing import POPCOUNT_LUT, hamming_distances


# ── Availability ──────────────────────────────────────────────────────

def test_native_available():
    """Native scan should compile on any system with gcc/cc."""
    # This will fail in exotic environments (no compiler) — that's
    # expected and informative, not a bug.
    assert AVAILABLE, (
        "Native scan did not compile. Ensure gcc or cc is installed."
    )


# ── Correctness: native matches LUT ──────────────────────────────────

def _lut_hamming(codes, query_code):
    """Reference LUT implementation — always uses NumPy path."""
    xor = np.bitwise_xor(codes, query_code[None, :])
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int64)


@pytest.mark.skipif(not AVAILABLE, reason="native scan not compiled")
class TestNativeCorrectness:
    """Verify native scan produces identical results to LUT."""

    def test_small_exact(self):
        """Small hand-verifiable case."""
        codes = np.array([[0xFF, 0x00], [0x00, 0xFF], [0xAA, 0x55]], dtype=np.uint8)
        query = np.array([0xFF, 0xFF], dtype=np.uint8)
        expected = _lut_hamming(codes, query)
        result = hamming_distances_native(codes, query)
        np.testing.assert_array_equal(result, expected)

    def test_self_distance_zero(self):
        """Distance to self must be 0."""
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 256, size=(100, 32), dtype=np.uint8)
        for i in range(min(10, len(codes))):
            d = hamming_distances_native(codes, codes[i])
            assert d[i] == 0, f"Self-distance at index {i} = {d[i]}, expected 0"

    def test_random_large(self):
        """Random corpus — native must match LUT exactly."""
        rng = np.random.default_rng(123)
        codes = rng.integers(0, 256, size=(10_000, 32), dtype=np.uint8)
        query = rng.integers(0, 256, size=32, dtype=np.uint8)
        expected = _lut_hamming(codes, query)
        result = hamming_distances_native(codes, query)
        np.testing.assert_array_equal(result, expected)

    def test_all_zeros(self):
        """All-zero corpus and query — distance must be 0 everywhere."""
        codes = np.zeros((50, 16), dtype=np.uint8)
        query = np.zeros(16, dtype=np.uint8)
        result = hamming_distances_native(codes, query)
        np.testing.assert_array_equal(result, np.zeros(50, dtype=np.int64))

    def test_all_ones(self):
        """All-0xFF — distance to self is 0."""
        codes = np.full((50, 16), 0xFF, dtype=np.uint8)
        query = np.full(16, 0xFF, dtype=np.uint8)
        result = hamming_distances_native(codes, query)
        np.testing.assert_array_equal(result, np.zeros(50, dtype=np.int64))

    def test_max_distance(self):
        """Complementary codes — distance = 8 * B."""
        B = 32
        codes = np.zeros((1, B), dtype=np.uint8)
        query = np.full(B, 0xFF, dtype=np.uint8)
        result = hamming_distances_native(codes, query)
        assert result[0] == 8 * B

    @pytest.mark.parametrize("B", [1, 3, 7, 8, 15, 16, 31, 32, 64, 96, 128])
    def test_various_widths(self, B):
        """Native must handle non-power-of-two byte widths (tail loop)."""
        rng = np.random.default_rng(B)
        codes = rng.integers(0, 256, size=(500, B), dtype=np.uint8)
        query = rng.integers(0, 256, size=B, dtype=np.uint8)
        expected = _lut_hamming(codes, query)
        result = hamming_distances_native(codes, query)
        np.testing.assert_array_equal(result, expected)

    def test_single_row(self):
        """Edge case: corpus with exactly one row."""
        codes = np.array([[0xAB, 0xCD]], dtype=np.uint8)
        query = np.array([0xAB, 0xCD], dtype=np.uint8)
        result = hamming_distances_native(codes, query)
        assert result.shape == (1,)
        assert result[0] == 0


# ── Integration: hamming_distances dispatches correctly ───────────────

@pytest.mark.skipif(not AVAILABLE, reason="native scan not compiled")
class TestDispatch:
    """Verify that packing.hamming_distances uses the native path."""

    def test_dispatch_matches_lut(self):
        """The dispatched function must match the pure-LUT reference."""
        rng = np.random.default_rng(999)
        codes = rng.integers(0, 256, size=(5_000, 32), dtype=np.uint8)
        query = rng.integers(0, 256, size=32, dtype=np.uint8)
        expected = _lut_hamming(codes, query)
        result = hamming_distances(codes, query)
        np.testing.assert_array_equal(result, expected)

    def test_search_uses_native(self):
        """End-to-end: SignBitQuantizer.search must work with native."""
        from remax import SignBitQuantizer
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 256))
        q = SignBitQuantizer(d=256, seed=1)
        codes = q.encode(X)
        idx, dist = q.search(X[0], codes, k=5, return_distances=True)
        assert idx[0] == 0, "Self should be first result"
        assert dist[0] == 0, "Self-distance should be 0"
