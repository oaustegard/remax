"""Tests for ``remax.corpus.Corpus``.

Required by issue #15:
  1. Build: packed index + SQLite metadata are written correctly.
  2. Search-with-metadata: ``Corpus.search()`` returns ``Result`` objects with
     the right record IDs and distances, in rank order.
  3. Reverse lookup: ``Corpus.lookup()`` maps record_id → array position.
  4. Round-trip: re-loading a corpus from disk gives the same search results.
  5. Missing-ID lookup returns None.
  6. Optional JSON metadata survives the round-trip.
  7. Error paths: wrong id count, wrong meta count, missing files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from remax.corpus import Corpus, Result


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _make_corpus(
    tmpdir,
    n: int = 50,
    d: int = 64,
    seed: int = 0,
    with_meta: bool = False,
) -> tuple[Corpus, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, d))
    ids = [f"doc-{i:04d}" for i in range(n)]
    meta = [{"idx": i, "score": float(i) / n} for i in range(n)] if with_meta else None
    c = Corpus.build(tmpdir, vectors, ids, d=d, seed=seed, meta=meta)
    return c, vectors, ids


# --------------------------------------------------------------------- #
# 1. Build — files written correctly
# --------------------------------------------------------------------- #

def test_build_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_corpus(tmpdir)
        assert (Path(tmpdir) / "index.bin").exists()
        assert (Path(tmpdir) / "meta.db").exists()


def test_build_properties():
    with tempfile.TemporaryDirectory() as tmpdir:
        n, d = 40, 128
        c, _, _ = _make_corpus(tmpdir, n=n, d=d)
        assert c.n == n
        assert c.d == d
        assert c.codes.shape == (n, d // 8)
        assert c.codes.dtype == np.uint8


# --------------------------------------------------------------------- #
# 2. Search-with-metadata
# --------------------------------------------------------------------- #

def test_search_returns_result_objects():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_corpus(tmpdir)
        results = c.search(vectors[0], k=5)
        assert len(results) == 5
        assert all(isinstance(r, Result) for r in results)


def test_search_self_is_first():
    """Self-retrieval: query vector[i] should find itself at rank 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_corpus(tmpdir, n=50, d=64, seed=7)
        for i in [0, 10, 49]:
            results = c.search(vectors[i], k=1)
            assert results[0].record_id == ids[i], (
                f"Expected {ids[i]} at rank 0, got {results[0].record_id}"
            )
            assert results[0].distance == 0


def test_search_distances_nondecreasing():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, _ = _make_corpus(tmpdir)
        results = c.search(vectors[0], k=10)
        dists = [r.distance for r in results]
        assert dists == sorted(dists)


def test_search_ranks_are_sequential():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, _ = _make_corpus(tmpdir)
        results = c.search(vectors[0], k=8)
        assert [r.rank for r in results] == list(range(8))


def test_search_record_ids_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_corpus(tmpdir)
        results = c.search(vectors[5], k=10)
        for r in results:
            assert r.record_id in ids


def test_search_k_larger_than_n_returns_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        n = 15
        c, vectors, _ = _make_corpus(tmpdir, n=n, d=64)
        results = c.search(vectors[0], k=100)
        assert len(results) == n


# --------------------------------------------------------------------- #
# 3. Reverse lookup
# --------------------------------------------------------------------- #

def test_lookup_known_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, ids = _make_corpus(tmpdir, n=30)
        for i in [0, 14, 29]:
            pos = c.lookup(ids[i])
            assert pos == i, f"Expected pos={i}, got {pos}"


def test_lookup_missing_id_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_corpus(tmpdir, n=10)
        assert c.lookup("nonexistent-id") is None


def test_lookup_all_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        n = 20
        c, _, ids = _make_corpus(tmpdir, n=n)
        for i, id_ in enumerate(ids):
            assert c.lookup(id_) == i


# --------------------------------------------------------------------- #
# 4. Round-trip: reload from disk
# --------------------------------------------------------------------- #

def test_reload_gives_same_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        c1, vectors, ids = _make_corpus(tmpdir, n=30, d=64, seed=3)
        c2 = Corpus(tmpdir)

        assert c2.n == c1.n
        assert c2.d == c1.d
        np.testing.assert_array_equal(c1.codes, c2.codes)

        r1 = c1.search(vectors[0], k=5)
        r2 = c2.search(vectors[0], k=5)
        assert [r.record_id for r in r1] == [r.record_id for r in r2]
        assert [r.distance for r in r1] == [r.distance for r in r2]


# --------------------------------------------------------------------- #
# 5. JSON metadata round-trip
# --------------------------------------------------------------------- #

def test_meta_survives_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_corpus(tmpdir, n=20, with_meta=True)
        results = c.search(vectors[0], k=1)
        r = results[0]
        assert r.meta is not None
        assert "idx" in r.meta
        assert "score" in r.meta
        # The top result is the self-match; its idx should equal its position
        expected_pos = c.lookup(r.record_id)
        assert r.meta["idx"] == expected_pos


def test_no_meta_gives_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_corpus(tmpdir, n=10, with_meta=False)
        results = c.search(vectors[0], k=3)
        for r in results:
            assert r.meta is None


# --------------------------------------------------------------------- #
# 6. Error paths
# --------------------------------------------------------------------- #

def test_build_wrong_id_count_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        with pytest.raises(ValueError, match="len\\(ids\\)"):
            Corpus.build(tmpdir, vectors, ["only-one-id"])


def test_build_wrong_meta_count_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        ids = [str(i) for i in range(10)]
        with pytest.raises(ValueError, match="len\\(meta\\)"):
            Corpus.build(tmpdir, vectors, ids, meta=[{"x": 1}])


def test_build_non_byte_aligned_d_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 7))
        ids = [str(i) for i in range(10)]
        with pytest.raises(ValueError, match="divisible by 8"):
            Corpus.build(tmpdir, vectors, ids)


def test_load_missing_bin_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="index.bin"):
            Corpus(tmpdir)


def test_load_missing_db_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create index.bin but no meta.db
        (Path(tmpdir) / "index.bin").write_bytes(b"\x00" * 25)
        with pytest.raises(FileNotFoundError, match="meta.db"):
            Corpus(tmpdir)


# --------------------------------------------------------------------- #
# 7. Repr
# --------------------------------------------------------------------- #

def test_repr():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_corpus(tmpdir, n=5, d=32)
        r = repr(c)
        assert "Corpus" in r
        assert "n=5" in r
        assert "d=32" in r


def test_repr_centered():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        ids = [f"doc-{i}" for i in range(10)]
        c = Corpus.build(tmpdir, vectors, ids, center=True)
        r = repr(c)
        assert "centered" in r


# --------------------------------------------------------------------- #
# 8. Centering — mean persistence and auto-centering
# --------------------------------------------------------------------- #

def _make_centered_corpus(
    tmpdir,
    n: int = 50,
    d: int = 64,
    seed: int = 0,
) -> tuple[Corpus, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, d)) + 5.0  # non-zero mean
    ids = [f"doc-{i:04d}" for i in range(n)]
    c = Corpus.build(tmpdir, vectors, ids, d=d, seed=seed, center=True)
    return c, vectors, ids


def test_center_saves_mean_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_centered_corpus(tmpdir)
        assert (Path(tmpdir) / "mean.npy").exists()


def test_no_center_no_mean_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir)
        assert not (Path(tmpdir) / "mean.npy").exists()


def test_centered_property():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_centered_corpus(tmpdir)
        assert c.centered is True
        assert c.mean is not None
        assert c.mean.shape == (64,)


def test_not_centered_property():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_corpus(tmpdir)
        assert c.centered is False
        assert c.mean is None


def test_mean_matches_corpus_mean():
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, _ = _make_centered_corpus(tmpdir)
        expected_mean = vectors.mean(axis=0)
        np.testing.assert_allclose(c.mean, expected_mean, rtol=1e-10)


def test_mean_is_copy():
    """Mutating the returned mean must not affect the corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c, _, _ = _make_centered_corpus(tmpdir)
        m = c.mean
        m[:] = 999.0
        np.testing.assert_array_less(c.mean, 900.0)  # not mutated


def test_centered_search_self_retrieval():
    """With centering, raw (un-centered) queries should still self-retrieve."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c, vectors, ids = _make_centered_corpus(tmpdir, n=50, d=64, seed=7)
        # Pass the RAW vector — search() should auto-center.
        for i in [0, 10, 49]:
            results = c.search(vectors[i], k=1)
            assert results[0].record_id == ids[i], (
                f"Expected {ids[i]} at rank 0, got {results[0].record_id}"
            )
            assert results[0].distance == 0


def test_centered_roundtrip():
    """Reload a centered corpus and verify mean + search are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c1, vectors, ids = _make_centered_corpus(tmpdir, n=30, d=64, seed=3)
        c2 = Corpus(tmpdir)

        assert c2.centered is True
        np.testing.assert_array_equal(c1.mean, c2.mean)

        r1 = c1.search(vectors[0], k=5)
        r2 = c2.search(vectors[0], k=5)
        assert [r.record_id for r in r1] == [r.record_id for r in r2]
        assert [r.distance for r in r1] == [r.distance for r in r2]


def test_center_false_overwrites_mean():
    """Rebuilding without centering should remove a stale mean.npy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_centered_corpus(tmpdir)
        assert (Path(tmpdir) / "mean.npy").exists()
        # Rebuild without centering
        _make_corpus(tmpdir)
        assert not (Path(tmpdir) / "mean.npy").exists()
