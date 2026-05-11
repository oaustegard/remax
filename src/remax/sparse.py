"""remax.sparse — sparse → sign-packed count-sketch encoder.

Generalizes :class:`remax.SignBitQuantizer` from dense Gaussian rotations
to a sparse input modality. The encoder is a signed count-sketch
(Charikar–Chen–Farach-Colton 2004) followed by sign-packing into
``uint8`` codes compatible with :func:`remax.hamming_distances`.

The high-leverage application is BM25 / TF-IDF / feature-hashed
bag-of-words / SPLADE-style learned-sparse embeddings — but this module
ships only the universal primitive. Application-specific wiring lives in
issues #2 and #4.

Mechanics
---------
For each input dimension ``j`` we draw two independent hashes from a
SplitMix64 stream salted by ``seed``:

* ``bucket[j] = H1(seed, j) mod k`` — the count-sketch column.
* ``sign[j]   = ±1`` from the parity of ``H2(seed, j)`` — the sign flip
  that turns the sketch into a *signed* count-sketch.

Concrete hash spec (SplitMix64; Steele, Lea, Flood, JCSS 2014):

    salt_a   = SplitMix64(seed)
    salt_b   = SplitMix64(seed XOR 0x9E3779B97F4A7C15)
    H1(s, j) = SplitMix64((salt_a + j) mod 2^64)
    H2(s, j) = SplitMix64((salt_b + j) mod 2^64)
    sign[j]  = +1 if (H2 AND 1) == 0 else -1

Per document row ``i`` we accumulate ``sign[j] * X[i, j]`` for every
nonzero ``j`` into a length-``k`` float buffer, then ``> 0`` → bit-pack
big-endian into ``k // 8`` bytes.

Hamming distance on the packed codes is a monotone ``(1 - 2θ/π)``
estimator of the cosine of the projected vectors (Goemans–Williamson
1995 applied to a fixed random hyperplane per sketch column), which
approximates the cosine of the originals up to count-sketch noise
(variance ∝ 1/k).

Centering
---------
``center=True`` subtracts the corpus mean before projection. Because the
sketch is linear, this is equivalent to projecting the mean once at
:meth:`fit` time and subtracting the resulting length-``k`` vector from
each row's buffer — so the per-row cost stays proportional to the row's
``nnz``, not to ``d``.

Whether centering helps on BM25-style sparse input is an empirical
question (issue #4 ablation). Default is ``center=False``.

Companion to remax
------------------
This is a Stage-1 *sketch-and-rank* primitive. It does not replace
:class:`remax.SignBitQuantizer` (which assumes dense, isotropic-friendly
input) — it provides a parallel path for sparse modalities so the rest
of the remax retrieval stack (``hamming_distances``, ``hamming_search``,
``Corpus``) is reachable from a sparse corpus without first densifying.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np

try:
    from scipy import sparse as _sp
except ImportError as e:  # pragma: no cover - scipy is a hard dep
    raise ImportError("remax.sparse requires scipy") from e

__all__ = ["SparseSignBitQuantizer"]

# SplitMix64 constants (Steele, Lea, Flood, JCSS 2014; the same algorithm
# used by OpenJDK's SplittableRandom).
_MASK64 = (1 << 64) - 1
_C0 = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E3B9
_C2 = 0x94D049BB133111EB


def _splitmix64_scalar(x: int) -> int:
    """SplitMix64 over a Python ``int``. Used for seed-level salts."""
    x = (x + _C0) & _MASK64
    x = ((x ^ (x >> 30)) * _C1) & _MASK64
    x = ((x ^ (x >> 27)) * _C2) & _MASK64
    return x ^ (x >> 31)


def _splitmix64_vec(x: np.ndarray) -> np.ndarray:
    """SplitMix64 over a ``uint64`` array. Used for per-dim hashes."""
    c0 = np.uint64(_C0)
    c1 = np.uint64(_C1)
    c2 = np.uint64(_C2)
    s30 = np.uint64(30)
    s27 = np.uint64(27)
    s31 = np.uint64(31)
    x = x + c0
    x = (x ^ (x >> s30)) * c1
    x = (x ^ (x >> s27)) * c2
    x = x ^ (x >> s31)
    return x


def _make_hashes(seed: int, d: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bucket[d] int64, sign[d] int8)`` for the spec'd hash."""
    seed64 = int(seed) & _MASK64
    salt_a = _splitmix64_scalar(seed64)
    salt_b = _splitmix64_scalar(seed64 ^ _C0)
    js = np.arange(d, dtype=np.uint64)
    h1 = _splitmix64_vec(np.uint64(salt_a) + js)
    h2 = _splitmix64_vec(np.uint64(salt_b) + js)
    bucket = (h1 % np.uint64(k)).astype(np.int64)
    parity_odd = (h2 & np.uint64(1)).astype(bool)
    sign = np.ones(d, dtype=np.int8)
    sign[parity_odd] = -1
    return bucket, sign


class SparseSignBitQuantizer:
    """Sparse → sign-packed count-sketch encoder.

    Parameters
    ----------
    d : int
        Input dimensionality (number of feature columns in ``X``).
    k : int
        Number of sketch bits. Must be a positive multiple of 8.
    seed : int, default=0
        Non-negative seed for the SplitMix64 hash stream. Same
        ``(d, k, seed)`` → byte-identical codes on the same input.
    center : bool, default=False
        If True, subtract the corpus mean (computed by :meth:`fit`)
        before sign-packing. See module docstring for the equivalence
        with projecting the mean once and subtracting per row.

    Attributes
    ----------
    d, k, seed, center
        As above.
    bucket_ : np.ndarray, shape (d,), dtype int64
        Per-input-dim sketch column index.
    sign_ : np.ndarray, shape (d,), dtype int8
        Per-input-dim sign in ``{-1, +1}``.
    mean_buf_ : np.ndarray | None, shape (k,)
        Projected corpus mean from :meth:`fit`. ``None`` until fit when
        ``center=True``; always ``None`` when ``center=False``.
    n_bits : int
        Code length in bits (``= k``), for symmetry with
        :class:`remax.SignBitQuantizer`.

    Examples
    --------
    >>> from scipy.sparse import random as sp_random
    >>> from remax.sparse import SparseSignBitQuantizer
    >>> X = sp_random(50, 10_000, density=0.01, format="csr")
    >>> enc = SparseSignBitQuantizer(d=10_000, k=128, seed=0)
    >>> codes = enc.encode(X)        # (50, 16) uint8
    """

    def __init__(
        self,
        d: int,
        k: int,
        seed: int = 0,
        center: bool = False,
    ):
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if k % 8 != 0:
            raise ValueError(
                f"k must be a multiple of 8 (got k={k}); remax codes "
                "are bit-packed into uint8 bytes."
            )
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(
                f"seed must be a non-negative integer, got {seed!r}"
            )
        self.d: int = int(d)
        self.k: int = int(k)
        self.seed: int = int(seed)
        self.center: bool = bool(center)
        self.bucket_, self.sign_ = _make_hashes(self.seed, self.d, self.k)
        self.mean_buf_: np.ndarray | None = None
        self.n_bits: int = self.k

    # ------------------------------------------------------------------ #
    # sklearn-style API
    # ------------------------------------------------------------------ #
    def fit(self, X) -> "SparseSignBitQuantizer":
        """Validate ``X`` and, if ``center=True``, compute the projected mean."""
        X = self._validate(X)
        if self.center:
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_buf = np.zeros(self.k, dtype=np.float64)
            np.add.at(
                mean_buf,
                self.bucket_,
                self.sign_.astype(np.float64) * mean,
            )
            self.mean_buf_ = mean_buf
        return self

    def encode(self, X) -> np.ndarray:
        """Encode ``(n, d)`` sparse input into ``(n, k // 8)`` uint8 codes."""
        X = self._validate(X)
        n = X.shape[0]
        buf = np.zeros((n, self.k), dtype=np.float64)
        if X.nnz > 0:
            indptr = X.indptr
            indices = X.indices
            data = X.data
            rows = np.repeat(
                np.arange(n, dtype=np.int64), np.diff(indptr)
            )
            cols = self.bucket_[indices]
            data_f = data.astype(np.float64, copy=False)
            contrib = self.sign_[indices] * data_f  # int8 * f64 → f64
            np.add.at(buf, (rows, cols), contrib)
        self._apply_centering(buf)
        return self._pack_signs(buf)

    def encode_query(self, q) -> np.ndarray:
        """Encode a single-row sparse matrix into a ``(k // 8,)`` code."""
        q = self._validate(q)
        if q.shape[0] != 1:
            raise ValueError(
                f"encode_query expects a single-row sparse matrix, "
                f"got shape {q.shape}"
            )
        return self.encode(q)[0]

    def encode_from_postings(
        self,
        postings: Iterable[
            tuple[Hashable, Iterable[tuple[Hashable, float]]]
        ],
        n: int,
        doc_id_map: Mapping[Hashable, int] | None = None,
    ) -> np.ndarray:
        """Encode an inverted-index stream into ``(n, k // 8)`` uint8 codes.

        Single-pass streaming construction over an inverted index — never
        materializes the equivalent ``(n, d)`` CSR. Wraps around term
        iterators from Elasticsearch / Lucene / FTS5 in production.

        Parameters
        ----------
        postings : iterable of (term, doc_weights)
            Iterator of ``(term, [(doc_id, weight), ...])`` pairs. ``term``
            is the integer column index in ``[0, d)``. ``doc_weights``
            may itself be a generator — it is consumed once.
        n : int
            Total number of documents (rows in the output).
        doc_id_map : mapping, optional
            ``{doc_id: row_index}`` for arbitrary hashable doc IDs. When
            absent, doc IDs must be integers in ``[0, n)``. A doc ID not
            present in the map raises :class:`KeyError`.

        Returns
        -------
        codes : np.ndarray, shape (n, k // 8), dtype uint8
            Byte-equal to :meth:`encode` on the equivalent CSR.

        Notes
        -----
        Terms with empty postings are silently skipped. Term order does
        not change the packed bytes (sums use float64 accumulators —
        exact for integer weights; bit-stable for typical real weights
        unless a bucket sum lands within ULPs of zero).
        """
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(
                f"n must be a non-negative integer, got {n!r}"
            )
        buf = np.zeros((int(n), self.k), dtype=np.float64)
        for term, doc_weights in postings:
            bucket, sign = self._hash_term(term)
            rows_list: list[int] = []
            weights_list: list[float] = []
            for doc_id, weight in doc_weights:
                row = (
                    doc_id_map[doc_id] if doc_id_map is not None else doc_id
                )
                rows_list.append(int(row))
                weights_list.append(float(weight))
            if not rows_list:
                continue
            rows_arr = np.asarray(rows_list, dtype=np.int64)
            weights_arr = np.asarray(weights_list, dtype=np.float64)
            np.add.at(buf, (rows_arr, bucket), sign * weights_arr)
        self._apply_centering(buf)
        return self._pack_signs(buf)

    # ------------------------------------------------------------------ #
    # Shared private helpers (extracted in #35 for the postings path).
    # ------------------------------------------------------------------ #
    def _hash_term(self, term: Hashable) -> tuple[int, int]:
        """Return ``(bucket, sign)`` for a single column-index term.

        ``term`` must be an integer in ``[0, d)`` — the streaming API
        treats terms as column indices into the same hash table that
        :meth:`encode` indexes with ``X.indices``. Callers wrapping a
        token-based inverted index are responsible for mapping tokens to
        column indices via their own vocabulary.
        """
        j = int(term)
        if j < 0 or j >= self.d:
            raise IndexError(
                f"term {term!r} maps to column {j}, out of range [0, {self.d})"
            )
        return int(self.bucket_[j]), int(self.sign_[j])

    def _pack_signs(self, buf: np.ndarray) -> np.ndarray:
        """Pack ``buf > 0`` into ``(..., k // 8)`` uint8 codes."""
        return np.packbits(buf > 0, axis=-1)

    def _apply_centering(self, buf: np.ndarray) -> None:
        """In-place subtraction of the projected corpus mean when ``center=True``."""
        if not self.center:
            return
        if self.mean_buf_ is None:
            raise RuntimeError(
                "center=True requires fit() before encode()"
            )
        buf -= self.mean_buf_[None, :]

    # ------------------------------------------------------------------ #
    def _validate(self, X):
        if not _sp.issparse(X):
            raise TypeError(
                f"X must be a scipy.sparse matrix, got "
                f"{type(X).__name__}"
            )
        X = X.tocsr()
        if X.shape[1] != self.d:
            raise ValueError(
                f"X has shape {X.shape}; expected (n, {self.d})."
            )
        return X
