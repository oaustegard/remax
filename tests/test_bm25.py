"""Tests for ``remax.bm25`` — BM25 weight utility for sparse pipelines.

Required by issue #34:

  1. Reference equivalence vs ``rank_bm25.BM25Okapi`` (ε ≤ 1e-6).
  2. ``k1=0`` → pure IDF (TF cancels).
  3. ``b=0`` → no length normalization.
  4. OOV query terms dropped silently.
  5. Empty doc → all-zero CSR row, no NaN.
  6. Vocabulary stability — same docs → same vocab dict (locked to sorted).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from remax.bm25 import bm25_csr, bm25_query


# A 10-doc corpus engineered so every term appears in ≤ 5/10 docs (df ≤ N/2).
# That keeps Okapi IDF non-negative, so rank_bm25's epsilon floor — which is a
# library-specific quirk, not part of BM25 — never fires and equivalence is exact.
CORPUS = [
    ["alpha", "beta", "gamma"],
    ["alpha", "delta", "epsilon"],
    ["beta", "gamma", "zeta"],
    ["delta", "eta", "theta"],
    ["epsilon", "zeta", "iota"],
    ["eta", "theta", "kappa"],
    ["iota", "kappa", "lambda"],
    ["lambda", "mu", "nu"],
    ["mu", "nu", "xi"],
    ["xi", "omicron", "pi"],
]


def _doc_freq(corpus, vocab):
    df = np.zeros(len(vocab), dtype=np.float64)
    for doc in corpus:
        for term in set(doc):
            df[vocab[term]] += 1
    return df


def test_reference_equivalence_against_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    k1, b = 1.2, 0.75
    weights, vocab = bm25_csr(CORPUS, k1=k1, b=b)
    okapi = rank_bm25.BM25Okapi(CORPUS, k1=k1, b=b)
    df = _doc_freq(CORPUS, vocab)
    queries = [
        ["alpha"],
        ["beta", "gamma"],
        ["mu", "nu", "xi"],
        ["alpha", "alpha"],  # repeated terms — count should additively contribute
    ]
    for qs in queries:
        q_vec = bm25_query(qs, vocab, df=df, N=len(CORPUS), k1=k1, b=b)
        ours = np.asarray((weights @ q_vec.T).todense()).ravel()
        theirs = okapi.get_scores(qs)
        np.testing.assert_allclose(
            ours, theirs, atol=1e-6, rtol=0, err_msg=f"query={qs}"
        )


def test_k1_zero_collapses_to_idf():
    weights, vocab = bm25_csr(CORPUS, k1=0.0, b=0.75)
    N = len(CORPUS)
    df = _doc_freq(CORPUS, vocab)
    expected_idf = np.log((N - df + 0.5) / (df + 0.5))
    # With k1=0: weight(t, d) = idf(t) * tf * 1 / (tf + 0) = idf(t) for tf > 0.
    coo = weights.tocoo()
    assert coo.nnz > 0
    for c, v in zip(coo.col, coo.data):
        assert v == pytest.approx(expected_idf[c], abs=1e-12)


def test_b_zero_no_length_normalization():
    k1 = 1.2
    weights, vocab = bm25_csr(CORPUS, k1=k1, b=0.0)
    N = len(CORPUS)
    df = _doc_freq(CORPUS, vocab)
    expected_idf = np.log((N - df + 0.5) / (df + 0.5))
    # With b=0: weight(t, d) = idf(t) * tf * (k1+1) / (tf + k1) — independent of |d|.
    # Every term in CORPUS occurs at most once per doc, so tf=1 wherever nonzero.
    coo = weights.tocoo()
    for c, v in zip(coo.col, coo.data):
        tf = 1.0
        expected = expected_idf[c] * tf * (k1 + 1) / (tf + k1)
        assert v == pytest.approx(expected, abs=1e-12)


def test_oov_query_terms_dropped_silently():
    _, vocab = bm25_csr(CORPUS)
    df = _doc_freq(CORPUS, vocab)
    q = bm25_query(
        ["alpha", "not_a_real_term", "beta", "also_oov"],
        vocab,
        df=df,
        N=len(CORPUS),
    )
    assert q.shape == (1, len(vocab))
    coo = q.tocoo()
    assert sorted(coo.col.tolist()) == sorted([vocab["alpha"], vocab["beta"]])
    assert np.all(coo.data == 1.0)


def test_empty_doc_yields_zero_row_no_nan():
    corpus = [["alpha", "beta"], [], ["alpha"]]
    weights, vocab = bm25_csr(corpus)
    assert weights.shape == (3, len(vocab))
    row1 = np.asarray(weights[1].todense()).ravel()
    assert np.all(row1 == 0.0)
    assert not np.any(np.isnan(weights.data))
    assert not np.any(np.isinf(weights.data))


def test_vocab_sorted_and_stable():
    _, vocab1 = bm25_csr(CORPUS)
    _, vocab2 = bm25_csr(CORPUS)
    assert vocab1 == vocab2
    keys = list(vocab1.keys())
    assert keys == sorted(keys), "vocab is locked to sorted order"


def test_shape_and_dtype():
    weights, vocab = bm25_csr(CORPUS)
    assert isinstance(weights, scipy.sparse.csr_matrix)
    assert weights.shape == (len(CORPUS), len(vocab))
    assert weights.dtype == np.float64
