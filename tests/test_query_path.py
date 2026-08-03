"""Query-path behaviours: batching, connection reuse, out=, contiguity.

The deep version of these properties lives in
``bench/gates/query_path_gate.py``, which anchors on a float sign-disagreement
reference and the Charikar collision identity, and proves itself by going red
under eight simulated defects. These are the fast unit-level equivalents, so
the contract is pinned by ``pytest`` too and a regression shows up on every
push rather than only in the gate job.
"""

from __future__ import annotations

import sqlite3
import threading
import warnings

import numpy as np
import pytest

import remax
from remax.packing import NonContiguousCodesWarning, as_codes, hamming_distances

D = 64
N = 300


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((N, D)).astype(np.float32)
    ids = [f"rec-{i:04d}" for i in range(N)]
    meta = [{"i": i} for i in range(N)]
    c = remax.Corpus.build(tmp_path / "c", X, ids, seed=5, meta=meta)
    yield c, X, ids
    c.close()


# ── batch search (m > 1) ──────────────────────────────────────────────────

def test_batch_search_returns_one_result_list_per_query(corpus):
    """A 2-D query used to raise sqlite3.ProgrammingError from the IN bind."""
    c, X, _ = corpus
    Q = X[:4]
    out = c.search(Q, k=5)
    assert isinstance(out, list) and len(out) == 4
    assert all(isinstance(row, list) and len(row) == 5 for row in out)
    assert all(isinstance(r, remax.Result) for row in out for r in row)


def test_batch_matches_per_query(corpus):
    c, X, _ = corpus
    Q = X[10:16]
    assert c.search(Q, k=7) == [c.search(Q[i], k=7) for i in range(len(Q))]


def test_batch_rows_are_distinct(corpus):
    """A batch that silently scored query 0 m times would return m equal rows."""
    c, X, _ = corpus
    out = c.search(X[:5], k=5)
    assert len({tuple(r.record_id for r in row) for row in out}) == 5


def test_single_query_shape_is_unchanged(corpus):
    """1-D in, flat list[Result] out — the pre-existing contract."""
    c, X, _ = corpus
    out = c.search(X[0], k=5)
    assert isinstance(out, list) and isinstance(out[0], remax.Result)


def test_batch_ranks_restart_per_row(corpus):
    c, X, _ = corpus
    for row in c.search(X[:3], k=6):
        assert [r.rank for r in row] == list(range(6))


def test_3d_query_is_rejected(corpus):
    c, X, _ = corpus
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        c.search(X[:8].reshape(2, 4, D), k=3)


def test_k_zero_shape_follows_the_query_rank(corpus):
    c, X, _ = corpus
    assert c.search(X[0], k=0) == []
    assert c.search(X[:3], k=0) == [[], [], []]


# ── connection reuse ──────────────────────────────────────────────────────

def test_one_connection_across_many_searches(corpus):
    c, X, _ = corpus
    for i in range(20):
        c.search(X[i], k=3)
    c.lookup("rec-0007")
    assert len(c._connections) == 1


def test_results_match_an_independent_sqlite_read(corpus):
    """Anchor: the database, read without going through Corpus."""
    c, X, ids = corpus
    results = c.search(X[3], k=8)
    con = sqlite3.connect(f"file:{c._db_path}?mode=ro", uri=True)
    try:
        for r in results:
            pos = ids.index(r.record_id)
            row = con.execute(
                "SELECT record_id FROM corpus_meta WHERE rowid = ?", (pos,)
            ).fetchone()
            assert row[0] == r.record_id
    finally:
        con.close()


def test_second_search_is_not_the_first_ones_metadata(corpus):
    """Caching rows instead of the connection would tie these together."""
    c, X, _ = corpus
    a = [r.record_id for r in c.search(X[0], k=5)]
    b = [r.record_id for r in c.search(X[100], k=5)]
    assert a != b


def test_close_is_idempotent_and_blocks_further_search(corpus):
    c, X, _ = corpus
    c.search(X[0], k=3)
    c.close()
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.search(X[0], k=3)
    with pytest.raises(RuntimeError, match="closed"):
        c.lookup("rec-0000")


def test_context_manager_closes(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, D)).astype(np.float32)
    with remax.Corpus.build(
        tmp_path / "c", X, [f"r{i}" for i in range(40)], seed=6
    ) as c:
        assert c.search(X[0], k=3)
    assert c._closed


def test_connections_are_per_thread(corpus):
    """Thread-local, so a sqlite3 object never crosses threads."""
    c, X, _ = corpus
    c.search(X[0], k=3)
    errors = []

    def worker():
        try:
            assert len(c.search(X[1], k=3)) == 3
        except Exception as exc:  # pragma: no cover - the failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(c._connections) == 4  # main + 3 workers


def test_search_does_not_hold_a_write_lock(corpus):
    """mode=ro: another connection must still be able to write."""
    c, X, _ = corpus
    c.search(X[0], k=3)
    writer = sqlite3.connect(c._db_path, timeout=2.0)
    try:
        writer.execute("CREATE TABLE IF NOT EXISTS probe (x INTEGER)")
        writer.execute("INSERT INTO probe VALUES (1)")
        writer.commit()
    finally:
        writer.close()


# ── out= preallocation ────────────────────────────────────────────────────

def test_out_matches_a_fresh_allocation(corpus):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    for i in (0, 7, 42):
        code = q.encode(X[i])
        fresh = hamming_distances(c.codes, code)
        buf = np.empty(N, dtype=np.int32)
        got = hamming_distances(c.codes, code, out=buf)
        assert got is buf
        np.testing.assert_array_equal(fresh, buf)


def test_out_is_fully_rewritten(corpus):
    """Nothing from a previous query may survive into the next."""
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    buf = np.full(N, -12345, dtype=np.int32)
    hamming_distances(c.codes, q.encode(X[0]), out=buf)
    assert not (buf == -12345).any()


def test_out_works_on_both_kernels(corpus):
    """Native and NumPy fallback must agree, with and without out=."""
    import remax._native as native

    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    code = q.encode(X[2])
    buf = np.empty(N, dtype=np.int32)
    with_native = hamming_distances(c.codes, code, out=buf).copy()
    prev = native.AVAILABLE
    native.AVAILABLE = False
    try:
        buf2 = np.empty(N, dtype=np.int32)
        fallback = hamming_distances(c.codes, code, out=buf2).copy()
    finally:
        native.AVAILABLE = prev
    np.testing.assert_array_equal(with_native, fallback)


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.empty(N + 1, dtype=np.int32), "shape"),
        (np.empty(N, dtype=np.int64), "int32"),
        (np.empty((N, 2), dtype=np.int32), "shape"),
    ],
)
def test_out_validates(corpus, bad, match):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with pytest.raises(ValueError, match=match):
        hamming_distances(c.codes, q.encode(X[0]), out=bad)


def test_out_rejects_a_readonly_buffer(corpus):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    buf = np.empty(N, dtype=np.int32)
    buf.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        hamming_distances(c.codes, q.encode(X[0]), out=buf)


def test_batch_search_reuses_one_buffer_but_gives_per_query_answers(corpus):
    """The m-loop shares a scratch buffer; results must not bleed across it."""
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    batched = q.search(X[:5], c.codes, k=6)
    for i in range(5):
        np.testing.assert_array_equal(batched[i], q.search(X[i], c.codes, k=6))


# ── contiguity: loud, not silently slow ───────────────────────────────────

def _strided(codes):
    view = np.ascontiguousarray(np.repeat(codes, 2, axis=0))[::2]
    assert not view.flags["C_CONTIGUOUS"]
    return view


def test_non_contiguous_codes_warn(corpus):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with pytest.warns(NonContiguousCodesWarning, match="not C-contiguous"):
        hamming_distances(_strided(c.codes), q.encode(X[0]))


def test_the_warning_names_the_size_being_copied(corpus):
    """A warning that does not say what it cost is easy to keep ignoring."""
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hamming_distances(_strided(c.codes), q.encode(X[0]))
    assert "MB" in str(caught[0].message)


def test_contiguous_codes_do_not_warn_or_copy(corpus):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NonContiguousCodesWarning)
        hamming_distances(c.codes, q.encode(X[0]))
    assert as_codes(c.codes) is c.codes  # no copy at all


def test_the_copy_is_still_correct(corpus):
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    code = q.encode(X[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonContiguousCodesWarning)
        strided_d = hamming_distances(_strided(c.codes), code)
    np.testing.assert_array_equal(strided_d, hamming_distances(c.codes, code))


def test_native_refuses_a_strided_buffer(corpus):
    """A raw pointer into a strided array would read unrelated memory."""
    import remax._native as native

    if not native.AVAILABLE:
        pytest.skip("native kernel unavailable on this box")
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with pytest.raises(ValueError, match="C-contiguous"):
        native.hamming_distances_native(_strided(c.codes), q.encode(X[0]))


def test_wrong_dtype_is_an_error_not_a_silent_cast(corpus):
    """The old ascontiguousarray(codes, dtype=uint8) turned a bug into a number."""
    c, X, _ = corpus
    q = remax.SignBitQuantizer(d=D, seed=5)
    with pytest.raises(ValueError, match="uint8"):
        hamming_distances(c.codes.astype(np.int64), q.encode(X[0]))
    with pytest.raises(ValueError, match="uint8"):
        as_codes(c.codes.astype(np.float32))


def test_as_codes_rejects_wrong_rank(corpus):
    c, _, _ = corpus
    with pytest.raises(ValueError, match="2-D"):
        as_codes(c.codes.ravel())
