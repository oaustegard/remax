"""Tests for security/correctness hardening of remax.

Covers the fixes from the 2026-05-03 adversarial review:
- _native.py: cache dir is private, atomic compile, ctypes int64 args,
  shape validation in hamming_distances_native.
- corpus.py: header magic + version, malformed-file rejection, k=0,
  large-k chunking, mode=rw refuses silent DB creation, file permissions.
"""
from __future__ import annotations

import os
import platform
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from remax import Corpus, SignBitQuantizer
from remax import _native
from remax.corpus import _HEADER_LEN, _MAGIC, _VERSION, _LEGACY_HEADER_LEN, _read_header


# ────────────────────── _native.py ─────────────────────────────────────────


@pytest.mark.skipif(not _native.AVAILABLE, reason="native scan unavailable")
def test_native_cache_dir_is_private(tmp_path, monkeypatch):
    """REMAX_CACHE_DIR honoured; default is *not* world-writable /tmp."""
    monkeypatch.setenv("REMAX_CACHE_DIR", str(tmp_path / "cache"))
    d = _native._cache_dir()
    assert d == tmp_path / "cache"
    # Mode bits: owner rwx only (0o700).
    if hasattr(os, "stat"):
        mode = os.stat(d).st_mode & 0o777
        assert mode == 0o700, f"cache dir mode {oct(mode)} should be 0o700"


@pytest.mark.skipif(not _native.AVAILABLE, reason="native scan unavailable")
def test_native_validates_query_shape():
    """Shape mismatches must raise, not segfault — kernel takes raw pointers."""
    rng = np.random.default_rng(0)
    codes = (rng.standard_normal((10, 16)) > 0).astype(np.uint8)
    codes = np.packbits(codes, axis=-1)  # (10, 2)

    # Wrong query rank
    with pytest.raises(ValueError, match="query_code must be 1-D"):
        _native.hamming_distances_native(codes, codes)
    # Wrong query length
    with pytest.raises(ValueError, match="does not match codes row length"):
        _native.hamming_distances_native(codes, np.zeros(99, dtype=np.uint8))
    # Wrong codes rank
    with pytest.raises(ValueError, match="codes must be 2-D"):
        _native.hamming_distances_native(
            codes.ravel(), np.zeros(2, dtype=np.uint8)
        )


@pytest.mark.skipif(not _native.AVAILABLE, reason="native scan unavailable")
def test_native_argtypes_are_int64():
    """Regression: ctypes c_int silently truncated n > 2^31 → garbage results."""
    import ctypes
    argtypes = _native._lib.hamming_scan.argtypes
    assert argtypes[3] is ctypes.c_int64
    assert argtypes[4] is ctypes.c_int64


@pytest.mark.skipif(not _native.AVAILABLE, reason="native scan unavailable")
def test_native_matches_python_lut():
    """Native and NumPy LUT paths must agree byte-for-byte across sizes."""
    rng = np.random.default_rng(42)
    for n, B in [(1, 1), (3, 8), (100, 32), (1000, 96)]:
        codes = rng.integers(0, 256, size=(n, B), dtype=np.uint8)
        query = rng.integers(0, 256, size=B, dtype=np.uint8)

        native = _native.hamming_distances_native(codes, query)

        # Python reference
        from remax.packing import POPCOUNT_LUT
        xor = np.bitwise_xor(codes, query[None, :])
        ref = POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int64)

        np.testing.assert_array_equal(native, ref)


# ────────────────────── corpus.py ──────────────────────────────────────────


@pytest.fixture
def small_corpus(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 32))
    ids = [f"r{i}" for i in range(50)]
    return Corpus.build(tmp_path / "c", X, ids, d=32, seed=7)


def test_corpus_writes_v1_magic(tmp_path, small_corpus):
    bin_path = Path(small_corpus._dir) / "index.bin"
    raw = bin_path.read_bytes()
    assert raw[:4] == _MAGIC
    assert raw[4] == _VERSION


def test_corpus_round_trip_v1(small_corpus):
    """Build → load → query path must round-trip cleanly with v1."""
    c2 = Corpus(small_corpus._dir)
    assert c2.n == small_corpus.n
    assert c2.d == small_corpus.d
    np.testing.assert_array_equal(c2.codes, small_corpus.codes)


def test_corpus_rejects_short_file(tmp_path):
    """A file smaller than even the legacy header is not a remax index."""
    d = tmp_path / "bad"
    d.mkdir()
    (d / "index.bin").write_bytes(b"\x00" * 10)
    (d / "meta.db").write_bytes(b"")
    with pytest.raises(ValueError, match="too short"):
        Corpus(d)


def test_corpus_rejects_bad_d(tmp_path):
    """d not divisible by 8 must be rejected (not silently truncated)."""
    d = tmp_path / "bad"
    d.mkdir()
    bad = bytearray(_HEADER_LEN + 100)
    bad[0:4] = _MAGIC
    bad[4] = _VERSION
    bad[5] = 0  # no seed
    bad[8:16] = (10).to_bytes(8, "little", signed=True)
    bad[16:24] = (33).to_bytes(8, "little", signed=True)  # 33 not /8
    (d / "index.bin").write_bytes(bytes(bad))
    (d / "meta.db").write_bytes(b"")
    with pytest.raises(ValueError, match="not divisible by 8"):
        Corpus(d)


def test_corpus_rejects_negative_n(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    bad = bytearray(_HEADER_LEN + 16)
    bad[0:4] = _MAGIC
    bad[4] = _VERSION
    bad[8:16] = (-1).to_bytes(8, "little", signed=True)
    bad[16:24] = (8).to_bytes(8, "little", signed=True)
    (d / "index.bin").write_bytes(bytes(bad))
    (d / "meta.db").write_bytes(b"")
    with pytest.raises(ValueError, match="n=-1"):
        Corpus(d)


def test_corpus_rejects_truncated_payload(tmp_path):
    """Header claims more codes than the file actually contains."""
    d = tmp_path / "bad"
    d.mkdir()
    # Claim n=1000 rows, d=8 → 1000 bytes payload, but only write 50.
    hdr = bytearray(_HEADER_LEN)
    hdr[0:4] = _MAGIC
    hdr[4] = _VERSION
    hdr[8:16] = (1000).to_bytes(8, "little", signed=True)
    hdr[16:24] = (8).to_bytes(8, "little", signed=True)
    (d / "index.bin").write_bytes(bytes(hdr) + b"\x00" * 50)
    (d / "meta.db").write_bytes(b"")
    with pytest.raises(ValueError, match="corrupt index"):
        Corpus(d)


def test_corpus_rejects_unsupported_version(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    hdr = bytearray(_HEADER_LEN)
    hdr[0:4] = _MAGIC
    hdr[4] = 99  # future version
    hdr[8:16] = (0).to_bytes(8, "little", signed=True)
    hdr[16:24] = (8).to_bytes(8, "little", signed=True)
    (d / "index.bin").write_bytes(bytes(hdr))
    (d / "meta.db").write_bytes(b"")
    with pytest.raises(ValueError, match="version 99"):
        Corpus(d)


def test_corpus_loads_legacy_v0_with_warning(tmp_path):
    """Files written before the magic-bytes hardening still load (with deprecation)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 32))
    n, d = X.shape
    seed = 13

    q = SignBitQuantizer(d=d, seed=seed)
    codes = q.encode(X)

    # Reproduce the old 25-byte header layout.
    old_header = bytearray(_LEGACY_HEADER_LEN)
    old_header[0:8] = (n).to_bytes(8, "little", signed=True)
    old_header[8:16] = (d).to_bytes(8, "little", signed=True)
    old_header[16] = 1
    old_header[17:25] = (seed).to_bytes(8, "little", signed=True)

    bin_dir = tmp_path / "legacy"
    bin_dir.mkdir()
    (bin_dir / "index.bin").write_bytes(bytes(old_header) + codes.tobytes())

    # Need a meta.db too — just an empty file works because we don't query it.
    import sqlite3
    con = sqlite3.connect(str(bin_dir / "meta.db"))
    con.executescript(
        "CREATE TABLE corpus_meta (rowid INTEGER PRIMARY KEY, "
        "record_id TEXT NOT NULL, meta TEXT);"
    )
    con.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = Corpus(bin_dir)
        assert any(
            issubclass(w.category, DeprecationWarning) and "v0" in str(w.message)
            for w in caught
        )
    assert c.n == n
    assert c.d == d


def test_corpus_search_k_zero(small_corpus):
    """k=0 must not raise an empty IN() SQL syntax error — return []."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal(small_corpus.d)
    assert small_corpus.search(q, k=0) == []


def test_corpus_search_large_k_chunks(tmp_path):
    """k > 999 must not blow SQLite's parameter limit."""
    rng = np.random.default_rng(0)
    n, d = 1500, 8
    X = rng.standard_normal((n, d))
    ids = [f"r{i}" for i in range(n)]
    c = Corpus.build(tmp_path / "big", X, ids, d=d, seed=1)
    # k=1500 forces > 999 placeholders in the IN clause; must succeed.
    res = c.search(X[0], k=n)
    assert len(res) == n
    assert {r.record_id for r in res} == set(ids)


def test_corpus_lookup_refuses_silent_db_creation(small_corpus):
    """Deleting meta.db underneath the loaded corpus must fail loudly."""
    db = Path(small_corpus._dir) / "meta.db"
    db.unlink()
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        small_corpus.lookup("r0")


def test_corpus_search_refuses_silent_db_creation(small_corpus):
    db = Path(small_corpus._dir) / "meta.db"
    db.unlink()
    rng = np.random.default_rng(0)
    q = rng.standard_normal(small_corpus.d)
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        small_corpus.search(q, k=5)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX file mode semantics",
)
def test_corpus_files_are_owner_only(small_corpus):
    """meta.db and index.bin should be 0o600; directory 0o700."""
    d = Path(small_corpus._dir)
    assert os.stat(d).st_mode & 0o777 == 0o700
    assert os.stat(d / "index.bin").st_mode & 0o777 == 0o600
    assert os.stat(d / "meta.db").st_mode & 0o777 == 0o600


def test_corpus_seed_persists_through_v1(tmp_path):
    """Various positive seeds round-trip through the v1 header."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 16))
    ids = [f"r{i}" for i in range(10)]

    # numpy.random.default_rng requires non-negative seeds, so we don't
    # test negatives — but the header still encodes int64-signed for
    # forward compat (e.g. if a future Generator accepts negative seeds).
    for seed in (None, 0, 1, 2**31, 2**62):
        c = Corpus.build(tmp_path / f"c{seed}", X, ids, d=16, seed=seed)
        c2 = Corpus(c._dir)
        assert c2._quantizer.seed == seed
