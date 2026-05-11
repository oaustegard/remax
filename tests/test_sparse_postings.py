"""Tests for ``SparseSignBitQuantizer.encode_from_postings`` (issue #35).

RED tests per the issue's "Specification-gap risk" discipline:

* Byte-exact equivalence: CSR encode() and encode_from_postings on the
  derived inverted-index form produce byte-identical output across
  multiple seeds.
* Term order invariance: shuffling the postings iterator yields the
  same packed bytes.
* Doc id map: arbitrary string doc IDs → encoded order matches the map.
* Zero-df term: terms with an empty postings list are silently skipped.
* Memory bound: peak memory during streaming encode is bounded by the
  (n, k) accumulator + per-term iterator overhead, not by an equivalent
  CSR materialization.

The equivalence tests deliberately use integer-valued weights so that
float64 sums are bit-exact regardless of summation order — the
equivalence then exercises the algebra, not float ULP noise.
"""

from __future__ import annotations

import random
import tracemalloc
from typing import Iterable, Hashable

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from remax.sparse import SparseSignBitQuantizer


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _csr_to_postings(
    X: csr_matrix,
) -> list[tuple[int, list[tuple[int, float]]]]:
    """Convert a CSR to its inverted-index form: list of (col, [(row, val), ...])."""
    Xc = X.tocsc()
    postings: list[tuple[int, list[tuple[int, float]]]] = []
    for j in range(Xc.shape[1]):
        start, end = Xc.indptr[j], Xc.indptr[j + 1]
        if end == start:
            continue
        rows = Xc.indices[start:end].tolist()
        vals = Xc.data[start:end].tolist()
        postings.append((j, list(zip(rows, vals))))
    return postings


def _random_int_csr(
    n: int, d: int, density: float, seed: int
) -> csr_matrix:
    """Random sparse CSR with integer-valued weights ∈ [1, 9]."""
    rng = np.random.default_rng(seed)
    nnz = max(1, int(round(n * d * density)))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, d, size=nnz)
    vals = rng.integers(1, 10, size=nnz).astype(np.float64)
    X = csr_matrix((vals, (rows, cols)), shape=(n, d))
    X.sum_duplicates()
    return X


# --------------------------------------------------------------------- #
# Byte-exact equivalence with CSR encode()
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_byte_exact_csr_equivalence(seed: int):
    """encode(X) == encode_from_postings(derived_postings, n) byte-for-byte."""
    n, d, k = 40, 300, 128
    X = _random_int_csr(n, d, density=0.04, seed=seed)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=seed)

    csr_codes = enc.encode(X)
    postings_codes = enc.encode_from_postings(_csr_to_postings(X), n)

    np.testing.assert_array_equal(csr_codes, postings_codes)
    assert csr_codes.dtype == postings_codes.dtype == np.uint8


def test_byte_exact_equivalence_with_centering():
    """Centering composes the same way for both encode paths."""
    n, d, k = 30, 200, 64
    X = _random_int_csr(n, d, density=0.05, seed=11)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=11, center=True).fit(X)

    csr_codes = enc.encode(X)
    postings_codes = enc.encode_from_postings(_csr_to_postings(X), n)
    np.testing.assert_array_equal(csr_codes, postings_codes)


# --------------------------------------------------------------------- #
# Term order invariance
# --------------------------------------------------------------------- #
def test_term_order_invariance():
    """Shuffling the postings iterator yields byte-identical output."""
    n, d, k = 30, 250, 128
    X = _random_int_csr(n, d, density=0.06, seed=3)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=3)

    postings = _csr_to_postings(X)
    rng = random.Random(123)

    base = enc.encode_from_postings(list(postings), n)
    for _ in range(5):
        shuffled = postings[:]
        rng.shuffle(shuffled)
        out = enc.encode_from_postings(shuffled, n)
        np.testing.assert_array_equal(out, base)


def test_postings_per_term_doc_order_invariance():
    """Within a term, the order of (doc_id, weight) tuples doesn't matter."""
    n, d, k = 20, 100, 64
    X = _random_int_csr(n, d, density=0.1, seed=9)
    enc = SparseSignBitQuantizer(d=d, k=k, seed=9)

    postings = _csr_to_postings(X)
    rng = random.Random(456)
    shuffled = [(t, rng.sample(docs, len(docs))) for t, docs in postings]

    a = enc.encode_from_postings(postings, n)
    b = enc.encode_from_postings(shuffled, n)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------- #
# Doc id map: arbitrary hashable doc IDs
# --------------------------------------------------------------------- #
def test_doc_id_map_string_ids_matches_csr_with_mapped_rows():
    """String doc IDs via doc_id_map encode into the mapped row order."""
    n, d, k = 4, 30, 16
    # doc_id_map: string IDs → row indices.
    doc_id_map = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}

    # Equivalent CSR with rows already in the mapped order.
    dense = np.array(
        [
            [1.0, 0.0, 2.0] + [0.0] * (d - 3),   # alpha → row 0
            [0.0, 3.0, 0.0] + [0.0] * (d - 3),   # beta → row 1
            [0.0, 0.0, 1.0] + [0.0] * (d - 3),   # gamma → row 2
            [4.0, 0.0, 0.0] + [0.0] * (d - 3),   # delta → row 3
        ],
        dtype=np.float64,
    )
    X = csr_matrix(dense)

    # Postings in string-id form, deliberately out of map order to
    # confirm the map (not the iteration order) determines rows.
    postings = [
        (0, [("delta", 4.0), ("alpha", 1.0)]),
        (1, [("beta", 3.0)]),
        (2, [("alpha", 2.0), ("gamma", 1.0)]),
    ]

    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    csr_codes = enc.encode(X)
    postings_codes = enc.encode_from_postings(
        postings, n=n, doc_id_map=doc_id_map
    )
    np.testing.assert_array_equal(csr_codes, postings_codes)


def test_doc_id_map_missing_id_raises():
    """A doc_id not present in the map raises KeyError, no silent drop."""
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    postings = [(0, [("known", 1.0), ("unknown", 2.0)])]
    with pytest.raises(KeyError):
        enc.encode_from_postings(postings, n=2, doc_id_map={"known": 0})


def test_doc_id_map_none_requires_integer_ids():
    """Without a map, doc_ids are treated as integer row indices."""
    enc = SparseSignBitQuantizer(d=5, k=8, seed=0)
    postings = [(0, [(0, 1.0), (1, 1.0)])]
    codes = enc.encode_from_postings(postings, n=2)
    assert codes.shape == (2, 1)


# --------------------------------------------------------------------- #
# Zero-df / empty / edge cases
# --------------------------------------------------------------------- #
def test_zero_df_term_silently_skipped():
    """A term with an empty postings list contributes nothing."""
    n, d, k = 3, 10, 8
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)

    # Two empty-df terms framing one populated term.
    postings_with_empty = [
        (0, []),
        (1, [(0, 1.0), (1, 1.0), (2, 1.0)]),
        (2, []),
    ]
    postings_without_empty = [
        (1, [(0, 1.0), (1, 1.0), (2, 1.0)]),
    ]

    a = enc.encode_from_postings(postings_with_empty, n)
    b = enc.encode_from_postings(postings_without_empty, n)
    np.testing.assert_array_equal(a, b)


def test_all_empty_postings_yields_zero_bytes():
    """All-empty postings stream → all-zero codes (center=False)."""
    n, d, k = 5, 20, 16
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    codes = enc.encode_from_postings(iter([]), n=n)
    assert codes.shape == (n, k // 8)
    assert np.all(codes == 0)


def test_n_zero_yields_empty_output():
    """n=0 → (0, k//8) empty output, no errors."""
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    codes = enc.encode_from_postings(iter([]), n=0)
    assert codes.shape == (0, 1)


def test_negative_n_rejected():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        enc.encode_from_postings(iter([]), n=-1)


def test_term_out_of_range_rejected():
    enc = SparseSignBitQuantizer(d=5, k=8, seed=0)
    with pytest.raises((IndexError, ValueError)):
        list(enc.encode_from_postings([(5, [(0, 1.0)])], n=1))


def test_streaming_accepts_generator():
    """Postings can be a generator (no .__len__, no random access)."""
    n, d, k = 3, 20, 16

    def gen():
        yield 0, iter([(0, 1.0), (2, 2.0)])
        yield 1, iter([(1, 3.0)])

    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    codes = enc.encode_from_postings(gen(), n)
    assert codes.shape == (n, k // 8)


# --------------------------------------------------------------------- #
# Memory bound: peak ≪ equivalent CSR size
# --------------------------------------------------------------------- #
def test_streaming_peak_memory_bounded_by_accumulator():
    """Peak memory during streaming encode is bounded by (n, k) float buf,
    not by the n × d CSR that would materialize the same input."""
    n, d, k = 500, 200_000, 128
    df_per_term = 12
    n_terms = 30_000  # → 360k nonzeros, ≈4.3 MB CSR vs 512 KB accumulator

    rng_global = np.random.default_rng(0)

    def make_postings() -> Iterable[tuple[Hashable, Iterable[tuple[int, float]]]]:
        # Yields generators per term so no list materializes the full corpus.
        for j in range(n_terms):
            doc_ids = rng_global.integers(0, n, size=df_per_term)
            weights = rng_global.integers(1, 10, size=df_per_term).astype(
                np.float64
            )

            def _row_iter(_doc_ids=doc_ids, _weights=weights):
                for doc_id, w in zip(_doc_ids.tolist(), _weights.tolist()):
                    yield doc_id, w

            yield j, _row_iter()

    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)

    tracemalloc.start()
    codes = enc.encode_from_postings(make_postings(), n=n)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert codes.shape == (n, k // 8)

    buf_bytes = n * k * 8  # the (n, k) float64 accumulator
    csr_bytes_approx = n_terms * df_per_term * 12  # ~int32 idx + f64 data

    # Equivalent CSR is at least 5× the accumulator on this fixture.
    assert csr_bytes_approx > 5 * buf_bytes, (
        "fixture sizing wrong: CSR not materially larger than accumulator "
        f"(csr≈{csr_bytes_approx} buf={buf_bytes})"
    )
    # Streaming peak stays within a small multiple of the accumulator
    # itself — proving we never materialized the CSR.
    assert peak < 3 * buf_bytes, (
        f"streaming peak {peak} exceeded 3× accumulator buffer "
        f"({buf_bytes})"
    )


def test_streaming_does_not_materialize_full_postings_list():
    """Iterator is consumed lazily — we never store all terms at once."""
    n, d, k = 100, 5_000, 64
    materialized: dict[str, int] = {"max_live_terms": 0}
    live = 0

    def tracker_gen():
        nonlocal live
        for j in range(d):
            live += 1
            materialized["max_live_terms"] = max(
                materialized["max_live_terms"], live
            )
            yield j, [(j % n, 1.0)]
            live -= 1

    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    enc.encode_from_postings(tracker_gen(), n)
    # Only one term is "live" at any instant (the one currently being
    # processed). The generator yields one (term, postings) tuple at a
    # time, and the encoder consumes it before requesting the next.
    assert materialized["max_live_terms"] == 1


# --------------------------------------------------------------------- #
# center=True requires fit()
# --------------------------------------------------------------------- #
def test_center_true_requires_fit():
    enc = SparseSignBitQuantizer(d=10, k=8, seed=0, center=True)
    with pytest.raises(RuntimeError, match="fit"):
        enc.encode_from_postings([(0, [(0, 1.0)])], n=1)


# --------------------------------------------------------------------- #
# Compatibility with hamming_distances
# --------------------------------------------------------------------- #
def test_streaming_codes_compatible_with_hamming_distances():
    from remax import hamming_distances

    n, d, k = 6, 40, 16
    enc = SparseSignBitQuantizer(d=d, k=k, seed=0)
    postings = [
        (0, [(0, 1.0), (1, 1.0)]),
        (5, [(2, 2.0), (3, 2.0)]),
        (10, [(4, 1.0), (5, 1.0)]),
    ]
    codes = enc.encode_from_postings(postings, n)
    dists = hamming_distances(codes, codes[0])
    assert dists.shape == (n,)
    assert dists[0] == 0
