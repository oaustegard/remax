"""remax.packing — bit-pack utilities and Hamming-distance scan.

Three primitives:

* :func:`encode_signs` — already-rotated floats → bit-packed ``uint8`` codes.
* :func:`hamming_distances` — broadcast XOR + popcount-LUT sum over a corpus.
* :func:`stable_top_k` — deterministic top-k over a distance/score vector.

When a C compiler is available, ``hamming_distances`` dispatches to a native
kernel using hardware ``POPCNT`` (~25–35× faster than the NumPy LUT fallback;
the ratio depends on ``n`` and ``d`` — :mod:`remax._native` carries the table).
The native path compiles automatically at first import and is cached; no extra
dependencies are required.  See :mod:`remax._native` for details.

Removed, 2026-08-03: ``hamming_search`` and ``asymmetric_search``
-----------------------------------------------------------------
Two single-query top-k wrappers used to live here and were exported from
``remax``. Nothing in the library ever called them: every search path
(:meth:`SignBitQuantizer.search`, ``SignBitQuantizer.search_asymmetric``, and
the stacked equivalents) composes ``hamming_distances`` / ``asymmetric_scores``
with ``stable_top_k`` directly, because it needs the batch loop and the
reusable output buffer the wrappers did not provide.

They were also the more dangerous of the two spellings: both took an
**already-rotated** query, so a caller passing a raw vector got plausible wrong
answers rather than an error, and neither accepted a batch. Use the quantizer
methods. If you genuinely want the functional form, it is one line each::

    order = stable_top_k(hamming_distances(codes, encode_signs(q @ R)), k)
    order = stable_top_k(-asymmetric_scores(q @ R, codes), k)
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _native

__all__ = [
    "POPCOUNT_LUT",
    "NonContiguousCodesWarning",
    "as_codes",
    "encode_signs",
    "hamming_distances",
    "stable_top_k",
]


class NonContiguousCodesWarning(UserWarning):
    """A code matrix had to be copied to be scanned.

    Its own class so callers can escalate it (``warnings.simplefilter("error",
    NonContiguousCodesWarning)``) or silence it deliberately, without touching
    every other warning remax might raise.
    """


def as_codes(codes: np.ndarray, *, argname: str = "codes") -> np.ndarray:
    """Validate a packed code matrix; copy only when unavoidable, and say so.

    The single place a code matrix is checked. There used to be three —
    ``core.SignBitQuantizer.search``, :func:`hamming_distances` and
    ``_native.hamming_distances_native`` each opened with
    ``np.ascontiguousarray(codes, dtype=np.uint8)``. On a well-formed index all
    three are no-ops, so the redundancy cost nothing and looked harmless. On a
    *strided view* it was three chances to copy the entire index, silently, on
    a path that runs once per query — and a strided view is not an exotic
    input. ``corpus.codes[::2]``, a subsampled evaluation set, a slice along
    the wrong axis: all produce one.

    Measured at 1M x 256 bits (32 MB): 5.6 ms contiguous, 16.6 ms through a
    strided view, every millisecond of the difference invisible.

    So: exactly one site may copy, and when it does it warns with the size.

    Raises
    ------
    ValueError
        If ``codes`` is not 2-D, or its dtype is not ``uint8``. A float or
        int64 array here is a caller error — packed codes are bytes — and the
        old silent ``dtype=np.uint8`` cast turned a wrong-array bug into wrong
        distances rather than an exception.

    Warns
    -----
    NonContiguousCodesWarning
        If a copy was required. The array is still copied and the call still
        succeeds; the point is that it stops being invisible.
    """
    arr = np.asarray(codes)
    if arr.ndim != 2:
        raise ValueError(f"{argname} must be 2-D, got ndim={arr.ndim}")
    if arr.dtype != np.uint8:
        raise ValueError(
            f"{argname} must be uint8 (packed bits), got dtype={arr.dtype}. "
            f"Encode with remax.encode_signs, or cast explicitly if you are "
            f"sure this array already holds packed bytes."
        )
    if not arr.flags["C_CONTIGUOUS"]:
        warnings.warn(
            f"{argname} is not C-contiguous; copying "
            f"{arr.nbytes / 1e6:.1f} MB to scan it. This happens on every "
            f"call. Hoist the copy with np.ascontiguousarray(codes) once, "
            f"outside your query loop.",
            NonContiguousCodesWarning,
            stacklevel=3,
        )
        arr = np.ascontiguousarray(arr)
    return arr


def _as_out(out: np.ndarray | None, n: int) -> np.ndarray:
    """Validate a caller-supplied output buffer, or allocate one."""
    if out is None:
        return np.empty(n, dtype=np.int32)
    if not isinstance(out, np.ndarray):
        raise TypeError(f"out must be a numpy array, got {type(out).__name__}")
    if out.shape != (n,):
        raise ValueError(
            f"out has shape {out.shape}; expected ({n},) to match the corpus"
        )
    if out.dtype != np.int32:
        raise ValueError(
            f"out must be int32 (the distance dtype), got {out.dtype}"
        )
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")
    if not out.flags["WRITEABLE"]:
        raise ValueError("out must be writeable")
    return out

# 256-entry byte-popcount lookup. uint16 is plenty (max value 8 per byte).
POPCOUNT_LUT: np.ndarray = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint16
)


def encode_signs(X_rotated: np.ndarray) -> np.ndarray:
    """Pack the sign bits of an already-rotated array into ``uint8`` bytes.

    Convention: ``x > 0`` → bit ``1``; ``x ≤ 0`` → bit ``0``. Bits are
    packed big-endian within each byte (numpy's default).

    Parameters
    ----------
    X_rotated : np.ndarray, shape (n, d) or (d,)
        Already-rotated input. Trailing dim ``d`` must be divisible by 8.

    Returns
    -------
    codes : np.ndarray, dtype uint8
        ``(n, d // 8)`` if input is 2-D; ``(d // 8,)`` if input is 1-D.
    """
    X = np.asarray(X_rotated)
    squeezed = False
    if X.ndim == 1:
        X = X[None, :]
        squeezed = True
    elif X.ndim != 2:
        raise ValueError(
            f"X_rotated must be 1-D or 2-D, got ndim={X.ndim}"
        )
    if X.shape[-1] % 8 != 0:
        raise ValueError(
            f"trailing dim must be divisible by 8 (got {X.shape[-1]}); "
            "remax codes are bit-packed into uint8 bytes."
        )
    bits = X > 0
    codes = np.packbits(bits, axis=-1)
    return codes[0] if squeezed else codes


def hamming_distances(
    codes: np.ndarray, query_code: np.ndarray, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Hamming distance from ``query_code`` to every row of ``codes``.

    Parameters
    ----------
    codes : np.ndarray, shape (n, B), dtype uint8
        Bit-packed corpus. Must be C-contiguous uint8; see :func:`as_codes`
        for what happens when it is not.
    query_code : np.ndarray, shape (B,), dtype uint8
        Bit-packed query.
    out : np.ndarray, shape (n,), dtype int32, keyword-only
        Destination buffer. When given, distances are written into it and it
        is returned, so a caller looping over queries allocates one ``(n,)``
        int32 array instead of one per query — at n=1M that is 4 MB of
        allocate-and-free per query. Omitting it preserves the previous
        behaviour exactly (a fresh array each call).

        The buffer is fully overwritten every call, so stale values from a
        previous query cannot survive into the next.

    Returns
    -------
    distances : np.ndarray, shape (n,), dtype int32
        Per-row Hamming distance, in ``[0, 8 * B]``. int32 is exact for any
        practical code width (8 * B overflows int32 only past a ~256 MB code)
        and keeps the downstream top-k argpartition narrow; ``search`` widens
        the returned top-k slice back to int64 for its public contract.
    """
    codes = as_codes(codes)
    q = np.ascontiguousarray(query_code, dtype=np.uint8)
    if q.ndim != 1 or q.shape[0] != codes.shape[1]:
        raise ValueError(
            f"query_code shape {q.shape} incompatible with "
            f"codes shape {codes.shape}"
        )
    out = _as_out(out, codes.shape[0])
    if _native.AVAILABLE:
        return _native.hamming_distances_native(codes, q, out=out)
    xor = np.bitwise_xor(codes, q[None, :])
    # POPCOUNT_LUT[xor] is (n, B) uint16 — popcount per byte. Sum across
    # bytes gives total Hamming distance per row.
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32, out=out)


def stable_top_k(dists: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` smallest distances, stably tie-broken by index.

    Equivalent to ``np.argsort(dists, kind="stable")[:k]`` but avoids the
    full O(n log n) sort when ``k`` ≪ ``n``: an :func:`numpy.argpartition`
    selects a candidate set, which is then widened to include all indices
    whose distance equals the kth-smallest value, and the candidate set is
    stably sorted.

    Why the widening: ``argpartition`` is unstable. When several distances
    tie at the kth value, it may keep a higher-indexed element inside the
    top-k partition and exclude a lower-indexed element with the same
    distance. Sorting only inside the partition cannot recover the
    lower-indexed element. Widening to ``dists <= pivot`` brings every
    tied candidate back into scope so the final stable sort matches
    ``argsort(dists, kind="stable")[:k]`` byte-for-byte.

    Parameters
    ----------
    dists : np.ndarray, shape (n,)
        Per-row distances. Any totally-ordered numeric dtype.
    k : int
        Number of indices to return. Must be positive; the result is
        clamped to ``min(k, n)``.

    Returns
    -------
    order : np.ndarray, shape (min(k, n),), dtype intp
        Indices into ``dists`` in ascending distance order, ties broken
        by ascending index.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = dists.shape[0]
    k_eff = min(k, n)
    if k_eff == n:
        return np.argsort(dists, kind="stable")[:k_eff]
    part = np.argpartition(dists, k_eff - 1)
    pivot = dists[part[k_eff - 1]]
    cand = np.flatnonzero(dists <= pivot)  # ascending order = stable for ties
    return cand[np.argsort(dists[cand], kind="stable")][:k_eff]


# ── asymmetric scoring ───────────────────────────────────────────────────────
#
# Hamming search binarizes BOTH sides. That is symmetric and cheap, but the
# query is a single vector per search -- it occupies no index storage -- so
# throwing away its precision buys nothing. Keeping it in float and scoring it
# against the +/-1 document bits is strictly more information at identical
# storage cost.
#
# Exa's web-scale index (exa.ai/blog/building-web-scale-vector-db) makes this
# choice, and measured on LFM2.5/SciFact it is worth +0.019 nDCG@10 at
# 128 B/vector, widening to +0.084 at 16 B -- the harder you compress, the more
# the query precision is carrying. See bench/results/lfm25_asymmetric.json.
#
# The obvious implementation unpacks the codes to a dense (n, d) +/-1 matrix and
# calls BLAS, but that allocates 32x the index and defeats the point. Instead
# precompute, per byte position, the partial dot product for all 256 possible
# byte values, then score by gather-and-sum. That is Exa's subvector lookup
# table at length 8 rather than length 4: O(n * d/8) table lookups instead of
# O(n * d) multiply-accumulates, with no dense intermediate.

_UNPACK_LUT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).astype(np.float32)  # (256, 8), MSB-first to match np.packbits


def asymmetric_scores(
    query_rotated: np.ndarray,
    codes: np.ndarray,
    *,
    chunk: int = 1 << 16,
) -> np.ndarray:
    """Dot product of a float query against sign-bit codes. Higher is better.

    Parameters
    ----------
    query_rotated : np.ndarray, shape (d,)
        Already-rotated query (caller applies ``query @ R``), NOT binarized.
    codes : np.ndarray, shape (n, d // 8), dtype uint8
        Bit-packed corpus from :func:`encode_signs`.
    chunk : int, keyword-only
        Rows scored per batch, bounding the (chunk, d/8) gather buffer.

    Returns
    -------
    scores : np.ndarray, shape (n,), dtype float32
        ``query . s`` where ``s`` is the corpus row decoded to +/-1.

    Notes
    -----
    Codes store ``b = (x > 0)`` as 0/1, while the value they represent is
    ``s = 2b - 1``. So ``q . s == 2 * (q . b) - sum(q)``. The table accumulates
    ``q . b``; the affine correction is applied once at the end. It does not
    change the ranking (``sum(q)`` is constant per query) but it makes the
    returned scores true inner products, so they stay meaningful to callers
    that threshold or fuse them rather than just sorting.
    """
    q = np.ascontiguousarray(query_rotated, dtype=np.float32).ravel()
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got ndim={codes.ndim}")
    if codes.dtype != np.uint8:
        raise ValueError(f"codes must be uint8, got {codes.dtype}")
    n_bytes = codes.shape[1]
    if q.size != n_bytes * 8:
        raise ValueError(
            f"query has {q.size} dims but codes carry {n_bytes * 8} bits."
        )

    # table[b, v] = partial dot product of the 8 dims in byte b against value v
    table = np.ascontiguousarray((_UNPACK_LUT @ q.reshape(n_bytes, 8).T).T)
    cols = np.arange(n_bytes)

    out = np.empty(codes.shape[0], dtype=np.float32)
    for start in range(0, codes.shape[0], chunk):
        block = codes[start : start + chunk]
        out[start : start + len(block)] = table[cols, block].sum(axis=1)
    return 2.0 * out - q.sum(dtype=np.float32)
