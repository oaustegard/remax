"""remax.packing — bit-pack utilities and Hamming-distance scan.

Three primitives:

* :func:`encode_signs` — already-rotated floats → bit-packed ``uint8`` codes.
* :func:`hamming_distances` — broadcast XOR + popcount-LUT sum over a corpus.
* :func:`hamming_search` — top-k by Hamming distance for a single rotated query.

When a C compiler is available, ``hamming_distances`` dispatches to a native
kernel using hardware ``POPCNT`` (~50–60× faster than the NumPy LUT fallback).
The native path compiles automatically at first import and is cached; no extra
dependencies are required.  See :mod:`remax._native` for details.
"""

from __future__ import annotations

import numpy as np

from . import _native

__all__ = [
    "POPCOUNT_LUT",
    "encode_signs",
    "hamming_distances",
    "hamming_search",
]

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
    codes: np.ndarray, query_code: np.ndarray
) -> np.ndarray:
    """Hamming distance from ``query_code`` to every row of ``codes``.

    Parameters
    ----------
    codes : np.ndarray, shape (n, B), dtype uint8
        Bit-packed corpus.
    query_code : np.ndarray, shape (B,), dtype uint8
        Bit-packed query.

    Returns
    -------
    distances : np.ndarray, shape (n,), dtype int64
        Per-row Hamming distance, in ``[0, 8 * B]``.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    q = np.ascontiguousarray(query_code, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got ndim={codes.ndim}")
    if q.ndim != 1 or q.shape[0] != codes.shape[1]:
        raise ValueError(
            f"query_code shape {q.shape} incompatible with "
            f"codes shape {codes.shape}"
        )
    if _native.AVAILABLE:
        return _native.hamming_distances_native(codes, q)
    xor = np.bitwise_xor(codes, q[None, :])
    # POPCOUNT_LUT[xor] is (n, B) uint16 — popcount per byte. Sum across
    # bytes gives total Hamming distance per row.
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int64)


def hamming_search(
    query_rotated: np.ndarray,
    codes: np.ndarray,
    k: int = 10,
    *,
    return_distances: bool = False,
):
    """Top-k Hamming search given an already-rotated query.

    Parameters
    ----------
    query_rotated : np.ndarray, shape (d,)
        Already-rotated query (caller is responsible for ``query @ R``).
    codes : np.ndarray, shape (n, d // 8), dtype uint8
        Bit-packed corpus produced by :func:`encode_signs`.
    k : int
        Number of neighbours to return.
    return_distances : bool, keyword-only
        If True, also return the Hamming distances of the top-k.

    Returns
    -------
    indices : np.ndarray, shape (k,), dtype intp
        Indices into ``codes``, sorted ascending by Hamming distance.
    distances : np.ndarray, shape (k,), dtype int64
        (Only if ``return_distances=True``.)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    q_code = encode_signs(np.asarray(query_rotated))
    if q_code.ndim != 1:
        raise ValueError("query_rotated must be a single 1-D vector")
    dists = hamming_distances(codes, q_code)
    n = dists.shape[0]
    k_eff = min(k, n)
    if k_eff == n:
        order = np.argsort(dists, kind="stable")[:k_eff]
    else:
        part = np.argpartition(dists, k_eff)[:k_eff]
        order = part[np.argsort(dists[part], kind="stable")]
    if return_distances:
        return order, dists[order]
    return order
