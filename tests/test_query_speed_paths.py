"""The fast query paths must be indistinguishable from the slow ones.

Four optimisations share one contract: **the output does not change**.

* threaded scan — row blocks dispatched into a pool (``threads=``)
* counting select — histogram top-k for integer distances
* blocked batch scan — corpus read once per block for all m queries
* mmap residency — ``Corpus(residency="mmap")``

Every test here compares a fast path against the slow path it replaces, or
against ``np.argsort(kind="stable")`` — which is what ``stable_top_k``
documents itself as, and what remax#32 was opened about. "Approximately the
same neighbours" is not the contract; byte-identical is.

The adversarial versions of these (a threading split that drops a block, a
counting select that takes a tie group from the wrong end, an mmap that
silently copies) live in ``bench/gates/query_path_gate.py``, which runs them
and confirms the gate goes red. These are the fast pytest-level equivalents.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import remax
from remax import packing
from remax.packing import (
    COUNTING_MIN_N,
    NonContiguousCodesWarning,
    counting_top_k,
    get_default_threads,
    hamming_distances,
    hamming_topk_batch,
    resolve_threads,
    set_default_threads,
    stable_top_k,
)

D = 128
N = 5000

#: Rows used by the threading tests. It has to exceed
#: ``_MIN_ROWS_PER_THREAD * 2``, or ``hamming_distances`` bypasses the pool and
#: the tests pass without ever running a thread — which is how the first draft
#: of this file "passed" the threading section at N=5000 (workers collapsed to
#: 1 every time). Codes are drawn directly rather than encoded, because these
#: tests are about the scan decomposition, not about the encoder.
N_THREADED = 4 * packing._MIN_ROWS_PER_THREAD + 137


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((N, D)).astype(np.float32)
    q = remax.SignBitQuantizer(d=D, seed=11)
    return X, q, q.encode(X)


@pytest.fixture
def big_codes():
    rng = np.random.default_rng(21)
    codes = rng.integers(0, 256, size=(N_THREADED, D // 8), dtype=np.uint8)
    query = rng.integers(0, 256, size=D // 8, dtype=np.uint8)
    return codes, query


@pytest.fixture(autouse=True)
def _restore_default_threads():
    """No test may leak a process-wide thread default into another."""
    before = get_default_threads()
    yield
    set_default_threads(before)


# ── 1. threaded scan ──────────────────────────────────────────────────────

def test_threading_is_off_by_default():
    """A library that silently spawns threads is a bad neighbour."""
    assert get_default_threads() == 1
    assert resolve_threads(None) == 1


def test_the_threading_fixture_actually_threads(big_codes):
    """Guard on the tests below, not on the library.

    Every "threaded vs serial" assertion in this file is vacuous if the
    small-n bypass collapses the worker count to 1. It did, in the first draft
    of this file, at N=5000 — the tests passed and tested nothing. This pins
    the fixture size to the bypass rule so the two cannot drift apart.
    """
    codes, query = big_codes
    n = codes.shape[0]
    assert min(4, max(1, n // packing._MIN_ROWS_PER_THREAD)) > 1
    packing._pools.clear()
    hamming_distances(codes, query, threads=4)
    assert packing._pools, "the pool was never used; the tests below are moot"


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 8, "auto"])
def test_threaded_scan_is_bit_identical(big_codes, threads):
    codes, query = big_codes
    serial = hamming_distances(codes, query, threads=1)
    np.testing.assert_array_equal(
        serial, hamming_distances(codes, query, threads=threads)
    )


@pytest.mark.parametrize("trim", [0, 1, 3, 137])
def test_threaded_scan_covers_every_row_when_n_is_not_divisible(big_codes, trim):
    """``n // T`` per block silently drops the ``n % T`` tail.

    A row count that no thread count divides, and a poisoned output buffer:
    a dropped block leaves its sentinel behind instead of raising, which is
    the whole reason this is checked rather than assumed.
    """
    codes, query = big_codes
    n = codes.shape[0] - trim
    codes = np.ascontiguousarray(codes[:n])
    for threads in (2, 3, 4, 7):
        poisoned = np.full(n, -777, dtype=np.int32)
        got = hamming_distances(codes, query, out=poisoned, threads=threads)
        assert not (got == -777).any(), f"a row was never written (T={threads})"
        np.testing.assert_array_equal(
            got, hamming_distances(codes, query, threads=1)
        )


def test_row_blocks_partition_exactly():
    """Disjoint, exhaustive, in order — the property threading rests on."""
    for n in (1, 2, 7, 4999, 1 << 20):
        for parts in (1, 2, 3, 4, 8, 17):
            blocks = packing._row_blocks(n, parts)
            assert blocks[0][0] == 0
            assert blocks[-1][1] == n
            for (_, prev_end), (start, _) in zip(blocks, blocks[1:]):
                assert start == prev_end
            assert sum(b - a for a, b in blocks) == n


def test_tiny_corpus_bypasses_the_pool(data):
    """Pool dispatch costs more than the scan below a few thousand rows."""
    X, q, _ = data
    codes = q.encode(X[:64])
    query = q.encode(X[0])
    packing._pools.clear()
    hamming_distances(codes, query, threads=4)
    assert not packing._pools, "a 64-row corpus should never touch the pool"


def test_batch_search_is_bit_identical_across_thread_counts(data):
    X, q, codes = data
    base = q.search(X[:6], codes, k=10, return_distances=True, threads=1)
    for t in (2, 4, "auto"):
        got = q.search(X[:6], codes, k=10, return_distances=True, threads=t)
        np.testing.assert_array_equal(base[0], got[0])
        np.testing.assert_array_equal(base[1], got[1])


def test_default_threads_setter_round_trips():
    assert set_default_threads(3) == 3
    assert get_default_threads() == 3
    assert resolve_threads(None) == 3
    assert resolve_threads(2) == 2
    assert set_default_threads("auto") >= 1


@pytest.mark.parametrize("bad", [0, -1, 2.5, None])
def test_bad_thread_counts_are_rejected(bad):
    if bad is None:
        pytest.skip("None means 'use the default', which is valid")
    with pytest.raises((ValueError, TypeError)):
        resolve_threads(bad)


def test_worker_exception_propagates(big_codes, monkeypatch):
    """A swallowed worker failure would return a half-written buffer."""
    codes, query = big_codes
    real = packing._scan_serial
    calls = {"n": 0}

    def boom(c, qq, out):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("kernel exploded")
        return real(c, qq, out)

    monkeypatch.setattr(packing, "_scan_serial", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        hamming_distances(codes, query, threads=4)


# ── 2. counting select ────────────────────────────────────────────────────

def _argsort_ref(d, k):
    return np.argsort(d, kind="stable")[:k]


@pytest.mark.parametrize("k", [1, 2, 5, 17, 100, 999])
def test_counting_select_matches_argsort_on_tie_dense_input(k):
    """Hamming distances at d=256 pile up around d/2 — ties are the rule."""
    rng = np.random.default_rng(2)
    d = rng.binomial(256, 0.5, size=20_000).astype(np.int32)
    np.testing.assert_array_equal(counting_top_k(d, k), _argsort_ref(d, k))


def test_counting_select_on_an_all_ties_array():
    """Every distance equal: the whole answer is the tie-break rule."""
    d = np.full(5000, 7, dtype=np.int32)
    np.testing.assert_array_equal(counting_top_k(d, 10), np.arange(10))
    np.testing.assert_array_equal(counting_top_k(d, 10), _argsort_ref(d, 10))


def test_counting_select_takes_the_tie_group_from_its_front():
    """The specific break this path must not commit.

    Nine values below the cutoff, then a tie group at the cutoff spanning the
    rest. The 10th slot must go to the LOWEST-indexed member of that group.
    Taking it from anywhere else gives the same k distances and different
    documents — the remax#32 failure, which is why it is spelled out rather
    than folded into the random test above.
    """
    d = np.full(2000, 5, dtype=np.int32)
    d[:9] = 1
    got = counting_top_k(d, 10)
    np.testing.assert_array_equal(got, np.arange(10))
    assert got[9] == 9, "the 10th slot must be the tie group's first member"
    np.testing.assert_array_equal(got, _argsort_ref(d, 10))


def test_counting_select_is_exhaustive_over_small_tie_dense_inputs():
    """Brute force: every (n, k, alphabet) combination against argsort."""
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 8, 33, 64):
        for hi in (1, 2, 3):
            for _ in range(6):
                d = rng.integers(0, hi + 1, size=n).astype(np.int32)
                for k in range(1, n + 1):
                    np.testing.assert_array_equal(
                        counting_top_k(d, k), _argsort_ref(d, k),
                        err_msg=f"n={n} hi={hi} k={k} d={d.tolist()}",
                    )


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16])
def test_counting_select_across_integer_dtypes(dtype):
    rng = np.random.default_rng(4)
    d = rng.integers(0, 30, size=3000).astype(dtype)
    np.testing.assert_array_equal(counting_top_k(d, 25), _argsort_ref(d, 25))


def test_counting_select_rejects_floats():
    """The float caller (search_asymmetric passes -scores) must not land here."""
    with pytest.raises(TypeError, match="integer"):
        counting_top_k(np.array([1.0, 2.0, 3.0]), 2)


def test_counting_select_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        counting_top_k(np.array([-1, 0, 3], dtype=np.int32), 2)


def test_a_wrong_value_bound_is_detected_not_trusted():
    """Too small a bound must not silently drop the rows above it."""
    d = np.arange(100, dtype=np.int32)
    with pytest.raises(ValueError):
        counting_top_k(d, 5, value_bound=10)
    np.testing.assert_array_equal(
        counting_top_k(d, 5, value_bound=99), _argsort_ref(d, 5)
    )


def test_stable_top_k_falls_back_rather_than_failing_on_negatives():
    """Dispatch is an optimisation; it must never change the answer."""
    rng = np.random.default_rng(5)
    d = rng.integers(-50, 50, size=COUNTING_MIN_N + 11).astype(np.int32)
    np.testing.assert_array_equal(stable_top_k(d, 12), _argsort_ref(d, 12))


def test_stable_top_k_takes_the_counting_path_above_the_threshold(monkeypatch):
    """The dispatch must actually fire, or the measurement means nothing."""
    seen = {}
    real = packing._counting_top_k

    def spy(dists, k, bound):
        seen["hit"] = True
        return real(dists, k, bound)

    monkeypatch.setattr(packing, "_counting_top_k", spy)
    rng = np.random.default_rng(6)
    big = rng.integers(0, 257, size=COUNTING_MIN_N + 1).astype(np.int32)
    stable_top_k(big, 10)
    assert seen.get("hit"), "counting path not reached above COUNTING_MIN_N"

    seen.clear()
    stable_top_k(big[:1000], 10)
    assert not seen, "counting path should not fire on a small array"

    seen.clear()
    stable_top_k(big.astype(np.float32), 10)
    assert not seen, "counting path must never take a float array"


def test_stable_top_k_agrees_with_itself_across_the_dispatch_threshold():
    """Same data, both implementations, same permutation."""
    rng = np.random.default_rng(7)
    d = rng.binomial(256, 0.5, size=COUNTING_MIN_N + 5000).astype(np.int32)
    counting = stable_top_k(d, 50)
    comparison = stable_top_k(d.astype(np.float64), 50).astype(np.intp)
    np.testing.assert_array_equal(counting, comparison)
    np.testing.assert_array_equal(counting, _argsort_ref(d, 50))


# ── 3. blocked batch scan ─────────────────────────────────────────────────

@pytest.mark.parametrize("block", [1, 2, 7, 64, 999, 5000, None])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_blocked_batch_is_bit_identical_to_the_per_query_loop(data, block, m):
    X, q, codes = data
    q_codes = q.encode(X[:m])
    idx, dist = hamming_topk_batch(codes, q_codes, 10, block=block)
    for i in range(m):
        d = hamming_distances(codes, q_codes[i])
        ref = stable_top_k(d, 10)
        np.testing.assert_array_equal(idx[i], ref)
        np.testing.assert_array_equal(dist[i], d[ref])


def test_blocked_batch_matches_argsort_on_a_tie_dense_corpus():
    """Duplicate rows: the merge across blocks is where index order breaks."""
    rng = np.random.default_rng(8)
    base = rng.integers(0, 256, size=(50, 16), dtype=np.uint8)
    codes = np.ascontiguousarray(np.tile(base, (40, 1)))  # 2000 rows, 40x dupes
    q_codes = rng.integers(0, 256, size=(4, 16), dtype=np.uint8)
    idx, _ = hamming_topk_batch(codes, q_codes, 25, block=37)
    for i in range(4):
        d = hamming_distances(codes, q_codes[i])
        np.testing.assert_array_equal(idx[i], np.argsort(d, kind="stable")[:25])


def test_blocked_batch_k_larger_than_a_block(data):
    """k > block size: no block can supply the whole answer on its own."""
    X, q, codes = data
    q_codes = q.encode(X[:3])
    idx, _ = hamming_topk_batch(codes, q_codes, 200, block=16)
    for i in range(3):
        d = hamming_distances(codes, q_codes[i])
        np.testing.assert_array_equal(idx[i], stable_top_k(d, 200))


def test_blocked_batch_k_larger_than_the_corpus(data):
    X, q, _ = data
    codes = q.encode(X[:9])
    idx, dist = hamming_topk_batch(codes, q.encode(X[:2]), 50, block=4)
    assert idx.shape == (2, 9) and dist.shape == (2, 9)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_blocked_and_threaded_together_stay_identical(data, threads):
    X, q, codes = data
    q_codes = q.encode(X[:6])
    ref, ref_d = hamming_topk_batch(codes, q_codes, 12, block=None, threads=1)
    got, got_d = hamming_topk_batch(codes, q_codes, 12, block=64,
                                    threads=threads)
    np.testing.assert_array_equal(ref, got)
    np.testing.assert_array_equal(ref_d, got_d)


def test_search_block_argument_does_not_change_the_answer(data):
    X, q, codes = data
    base = q.search(X[:8], codes, k=15, return_distances=True)
    for block in (13, 256, 100_000):
        got = q.search(X[:8], codes, k=15, return_distances=True, block=block)
        np.testing.assert_array_equal(base[0], got[0])
        np.testing.assert_array_equal(base[1], got[1])


def test_block_must_be_positive(data):
    _, q, codes = data
    with pytest.raises(ValueError, match="block"):
        hamming_topk_batch(codes, q.encode(np.zeros((1, D), np.float32)),
                           5, block=0)


def test_auto_block_does_not_block_a_single_query(data):
    """One query reads the corpus once regardless; blocking only adds work."""
    assert packing._resolve_block(None, 10**7, 1, 32) == 10**7
    assert packing._resolve_block(None, 10**7, 8, 32) < 10**7


# ── 4. mmap residency ─────────────────────────────────────────────────────

@pytest.fixture
def built(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.standard_normal((400, D)).astype(np.float32)
    ids = [f"rec-{i:04d}" for i in range(400)]
    meta = [{"i": i} for i in range(400)]
    c = remax.Corpus.build(tmp_path / "c", X, ids, seed=12, meta=meta)
    c.close()
    return tmp_path / "c", X, ids


def test_residency_defaults_to_load(built):
    path, _, _ = built
    with remax.Corpus(path) as c:
        assert c.residency == "load"
        assert not isinstance(c.codes, np.memmap)


def test_mmap_codes_are_actually_mapped(built):
    path, _, _ = built
    with remax.Corpus(path, residency="mmap") as c:
        assert c.residency == "mmap"
        assert isinstance(c.codes, np.memmap)


def test_mmap_codes_are_c_contiguous(built):
    """The 378 ms/query cliff this argument exists to not build.

    ``as_codes`` copies a non-contiguous code matrix — once per query, whole
    index — so an mmap whose payload offset broke contiguity would trade a
    one-off load for a permanent per-query copy, and the open-time benchmark
    would still look like a win.
    """
    path, _, _ = built
    with remax.Corpus(path, residency="mmap") as c:
        assert c.codes.flags["C_CONTIGUOUS"]
        assert packing.as_codes(c.codes) is not None


def test_mmap_search_does_not_copy_the_index(built):
    """The contiguity guard from #63 must stay silent on the mmap path."""
    path, X, _ = built
    with remax.Corpus(path, residency="mmap") as c:
        with warnings.catch_warnings():
            warnings.simplefilter("error", NonContiguousCodesWarning)
            c.search(X[:4], k=5)


def test_mmap_and_load_give_identical_results(built):
    path, X, _ = built
    with remax.Corpus(path) as a, remax.Corpus(path, residency="mmap") as b:
        np.testing.assert_array_equal(a.codes, b.codes)
        assert a.search(X[:5], k=7) == b.search(X[:5], k=7)
        assert a.n == b.n and a.d == b.d and a.rotation == b.rotation


def test_mmap_is_read_only(built):
    path, _, _ = built
    with remax.Corpus(path, residency="mmap") as c:
        with pytest.raises(ValueError):
            c.codes[0, 0] = 255


def test_unknown_residency_is_rejected(built):
    path, _, _ = built
    with pytest.raises(ValueError, match="residency"):
        remax.Corpus(path, residency="mmapp")


def test_empty_corpus_opens_under_both_residencies(tmp_path):
    """mmap of a zero-length region is an error; an empty corpus is not."""
    c = remax.Corpus.build(
        tmp_path / "e", np.empty((0, D), dtype=np.float32), [], seed=1
    )
    c.close()
    for residency in ("load", "mmap"):
        with remax.Corpus(tmp_path / "e", residency=residency) as got:
            assert got.n == 0
            assert got.codes.shape == (0, D // 8)


def test_repr_names_a_non_default_residency(built):
    path, _, _ = built
    with remax.Corpus(path, residency="mmap") as c:
        assert "residency='mmap'" in repr(c)
    with remax.Corpus(path) as c:
        assert "residency" not in repr(c)


# ── 5. asymmetric on Corpus, and the hoisted table ────────────────────────

def test_corpus_asymmetric_matches_the_quantizer(built):
    """The +0.019 nDCG@10 path, reachable through the only supported API."""
    path, X, ids = built
    with remax.Corpus(path) as c:
        got = c.search(X[:4], k=6, asymmetric=True)
        q = remax.SignBitQuantizer(d=D, seed=12)
        want = q.search_asymmetric(X[:4], c.codes, k=6)
        for i in range(4):
            assert [r.record_id for r in got[i]] == [ids[j] for j in want[i]]


def test_corpus_asymmetric_defaults_off(built):
    path, X, _ = built
    with remax.Corpus(path) as c:
        assert c.search(X[:3], k=5) == c.search(X[:3], k=5, asymmetric=False)


def test_hoisted_asymmetric_table_is_bit_identical(built):
    """The hoisted table build must not move a single ulp: ties would reorder.

    Compared against the loop body it replaced — ``asymmetric_scores`` once per
    query — fed the *same* rotated matrix.

    It is deliberately not compared against ``search_asymmetric(X[i])`` one
    query at a time, and that is worth writing down because the first version
    of this test did exactly that and failed. ``query @ rotation_`` is an
    ``(m, d) @ (d, d)`` GEMM; BLAS selects a different kernel for m=1 than for
    m=8, so ``rotated`` itself differs in the last ulp between the two shapes.
    Those float scores then differ by ~1e-6. That is **pre-existing** — it is
    true of this method on origin/main, verified before the test was
    rewritten — and it has nothing to do with hoisting the table. A test that
    conflates the two would have blocked a correct change and pointed at the
    wrong line.
    """
    path, X, _ = built
    q = remax.SignBitQuantizer(d=D, seed=12)
    with remax.Corpus(path) as c:
        got, got_scores = q.search_asymmetric(
            X[:8], c.codes, k=9, return_scores=True
        )
        rotated = np.asarray(X[:8], dtype=q.dtype) @ q.rotation_
        for i in range(8):
            scores = packing.asymmetric_scores(rotated[i], c.codes)
            order = packing.stable_top_k(-scores, 9)
            np.testing.assert_array_equal(got[i], order)
            np.testing.assert_array_equal(got_scores[i], scores[order])


def test_asymmetric_tables_match_the_per_query_build(built):
    path, X, _ = built
    q = remax.SignBitQuantizer(d=D, seed=12)
    rotated = np.asarray(X[:5], dtype=np.float32) @ q.rotation_
    tables, offsets = packing.asymmetric_tables(rotated, D // 8)
    for i in range(5):
        one, one_off = packing.asymmetric_tables(rotated[i], D // 8)
        np.testing.assert_array_equal(tables[i], one[0])
        assert offsets[i] == one_off[0]


def test_asymmetric_query_grouping_does_not_change_the_answer(built,
                                                              monkeypatch):
    """Force a tiny table budget so the group loop runs more than once."""
    import remax.core as core_mod

    path, X, _ = built
    q = remax.SignBitQuantizer(d=D, seed=12)
    with remax.Corpus(path) as c:
        ref = q.search_asymmetric(X[:7], c.codes, k=5)
        monkeypatch.setattr(core_mod, "_ASYM_TABLE_BUDGET", 1)
        np.testing.assert_array_equal(
            ref, q.search_asymmetric(X[:7], c.codes, k=5)
        )
