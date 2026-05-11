"""BM25 weight utility for sparse input pipelines.

Pure-function BM25 (Okapi variant) producing ``scipy.sparse.csr_matrix``
outputs suitable for feeding into a sparse sign-bit encoder (``#33``).

Not in the runtime critical path — convenience for benchmarks and examples.
Production users typically already have BM25 weights from
Elasticsearch / Solr / Lucene / FTS5.

Formula (Okapi BM25):

    weight(t, d) = idf(t) * tf(t,d) * (k1 + 1)
                 / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t)      = log( (N - df(t) + 0.5) / (df(t) + 0.5) )

The canonical log term can go negative for terms appearing in more than
half the corpus. ``rank_bm25`` clamps those to an ``epsilon * mean_idf``
floor; that is a search-engine convention, not part of BM25, and is not
applied here. Use Lucene's ``log(1 + ...)`` form (or filter high-DF
terms upstream) if non-negativity matters for your application.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse


def bm25_csr(
    documents: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[scipy.sparse.csr_matrix, dict[str, int]]:
    """Compute BM25 doc weights for a tokenized corpus.

    Parameters
    ----------
    documents
        Pre-tokenized documents. No further normalization is applied —
        casing / stemming / stopword removal happen upstream.
    k1, b
        BM25 hyperparameters.

    Returns
    -------
    weights : csr_matrix, shape ``(N, V)``, dtype ``float64``
        ``weights[i, vocab[t]]`` is the BM25 weight of term ``t`` in
        document ``i``. Zero entries are not stored.
    vocab : dict[str, int]
        Sorted-alphabetical mapping from term to column index. Vocab
        order is stable across runs given the same input.
    """
    N = len(documents)
    if N == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64), {}

    terms: set[str] = set()
    for doc in documents:
        terms.update(doc)
    vocab = {term: i for i, term in enumerate(sorted(terms))}
    V = len(vocab)
    if V == 0:
        return scipy.sparse.csr_matrix((N, 0), dtype=np.float64), vocab

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    doc_lens = np.zeros(N, dtype=np.float64)
    for i, doc in enumerate(documents):
        doc_lens[i] = len(doc)
        counts: dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        for term, c in counts.items():
            rows.append(i)
            cols.append(vocab[term])
            data.append(float(c))

    if not data:
        return scipy.sparse.csr_matrix((N, V), dtype=np.float64), vocab

    tf = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=(N, V), dtype=np.float64
    )
    df = np.asarray((tf != 0).sum(axis=0)).ravel().astype(np.float64)

    avgdl = doc_lens.mean()
    avgdl_safe = avgdl if avgdl > 0 else 1.0

    idf = np.log((N - df + 0.5) / (df + 0.5))

    tf_vals = tf.data
    row_idx = tf.row
    col_idx = tf.col
    dl = doc_lens[row_idx]
    numer = tf_vals * (k1 + 1.0)
    denom = tf_vals + k1 * (1.0 - b + b * dl / avgdl_safe)
    weights_data = idf[col_idx] * numer / denom

    out = scipy.sparse.csr_matrix(
        (weights_data, (row_idx, col_idx)),
        shape=(N, V),
        dtype=np.float64,
    )
    return out, vocab


def bm25_query(
    query_terms: list[str],
    vocab: dict[str, int],
    df: np.ndarray,
    N: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> scipy.sparse.csr_matrix:
    """Build a sparse query vector compatible with ``bm25_csr`` doc weights.

    Returns a ``(1, len(vocab))`` count vector where entries equal the
    number of occurrences of each in-vocab term in ``query_terms``.
    Because the full BM25 weighting (IDF + TF normalization) is baked
    into the doc-side matrix, the query side only needs to indicate
    which terms (and how many times) to sum over — i.e.
    ``weights @ query.T`` recovers ``BM25Okapi.get_scores`` exactly.

    OOV terms are silently dropped.

    The ``df``, ``N``, ``k1``, ``b`` parameters are accepted for
    signature symmetry with ``bm25_csr`` and to leave room for future
    query-side weighting variants. They are unused in v0.1.0.
    """
    del df, N, k1, b  # reserved for future query-side variants
    V = len(vocab)

    counts: dict[int, int] = {}
    for t in query_terms:
        j = vocab.get(t)
        if j is None:
            continue
        counts[j] = counts.get(j, 0) + 1

    if not counts:
        return scipy.sparse.csr_matrix((1, V), dtype=np.float64)

    cols = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    data = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    rows = np.zeros(len(cols), dtype=np.int64)
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(1, V), dtype=np.float64
    )


__all__ = ["bm25_csr", "bm25_query"]
