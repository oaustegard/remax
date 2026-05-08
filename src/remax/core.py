"""remax.core — 1-bit cosine LSH (Charikar 2002 SimHash).

Rotate inputs into an isotropic frame, take the sign of each coordinate,
pack the signs into bits. Hamming distance on the packed codes is — in
expectation — a monotone function of the angle between the original
vectors:

    Pr[ sign(<r, x>) ≠ sign(<r, y>) ] = θ(x, y) / π

(Goemans–Williamson 1995). On a *single* random direction this is a
binary collision indicator; on ``d`` independent directions stacked
together the empirical collision rate concentrates around ``θ/π`` with
variance ``∝ 1/d``, giving a rank-correct similarity estimate.

This module exposes both a class API (:class:`SignBitQuantizer`) and the
functional primitives it composes. They are equivalent; the class just
binds a fixed rotation and convenience methods around the functional core.

Companion to remex
------------------
remex's ``IVFCoarseIndex`` uses SimHash for *cell routing*; remax uses
stacked SimHash for *scoring*. The two compose orthogonally — a remax
Stage 1 + remex Stage 2 is the natural two-stage retrieval pipeline.
v0.1.0 ships the building blocks, not the integration.
"""

from __future__ import annotations

import numpy as np

# Re-export the functional API at this layer so callers can import either
# ``from remax.core import haar_rotation`` or ``from remax import haar_rotation``.
from .packing import encode_signs, hamming_distances, hamming_search
from .rotation import haar_rotation

__all__ = [
    "SignBitQuantizer",
    "haar_rotation",
    "encode_signs",
    "hamming_distances",
    "hamming_search",
]


class SignBitQuantizer:
    """1-bit cosine LSH quantizer (Charikar 2002 SimHash).

    Parameters
    ----------
    d : int
        Input dimension. Must be divisible by 8 (codes are packed bytewise).
    seed : int | None, default=None
        RNG seed for the Haar rotation. Same seed + same ``dtype`` →
        byte-identical codes.
    dtype : numpy dtype, default=np.float32
        Working precision for the rotation matrix and the matmul. The
        output is always 1 bit per dimension, so f64 precision in the
        rotation is wasted bandwidth — f32 halves memory traffic for the
        encode matmul (sgemm vs dgemm) on every BLAS, and on Apple
        silicon enables Accelerate's f32 path. Pass ``np.float64`` for
        bit-exact compatibility with corpora encoded before this default
        changed; recall is statistically identical either way.

    Attributes
    ----------
    d : int
        Input dimension.
    seed : int | None
        RNG seed used for the rotation.
    dtype : numpy dtype
        Working precision (matches ``rotation_.dtype``).
    rotation_ : np.ndarray, shape (d, d)
        Orthogonal rotation matrix. Established at construction time;
        ``fit`` is a no-op kept for sklearn-style ergonomics.
    n_bits : int
        Number of bits per code. Equal to ``d`` for the 1-bit quantizer.

    Examples
    --------
    >>> import numpy as np
    >>> from remax import SignBitQuantizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((1000, 768))
    >>> q = SignBitQuantizer(d=768, seed=42)
    >>> codes = q.encode(X)                    # (1000, 96) uint8
    >>> top = q.search(X[0], codes, k=10)       # (10,) intp
    >>> top[0] == 0                             # self at distance 0
    True
    """

    def __init__(
        self,
        d: int,
        seed: int | None = None,
        *,
        dtype: np.dtype | type = np.float32,
    ):
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        if d % 8 != 0:
            raise ValueError(
                f"d must be divisible by 8 (got d={d}); remax codes are "
                "bit-packed into uint8 bytes."
            )
        self.d: int = int(d)
        self.seed: int | None = seed
        self.dtype: np.dtype = np.dtype(dtype)
        self.rotation_: np.ndarray = haar_rotation(
            self.d, seed=seed, dtype=self.dtype
        )
        self.n_bits: int = self.d

    # ------------------------------------------------------------------ #
    # sklearn-style API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray | None = None) -> "SignBitQuantizer":
        """No-op fit, retained for API symmetry with sklearn-style transformers.

        The rotation is fully determined by ``(d, seed)`` and was established
        in ``__init__``. If ``X`` is given, its column count is validated
        against ``self.d`` and a ``ValueError`` is raised on mismatch.
        """
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != self.d:
                raise ValueError(
                    f"X has shape {X.shape}; expected (n, {self.d})."
                )
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode ``(n, d)`` float input into ``(n, d // 8)`` uint8 codes.

        Accepts a 1-D vector ``(d,)`` as a single-row batch; in that case
        the returned shape is ``(d // 8,)``.
        """
        X = np.asarray(X, dtype=self.dtype)
        squeezed = False
        if X.ndim == 1:
            X = X[None, :]
            squeezed = True
        elif X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got ndim={X.ndim}")
        if X.shape[1] != self.d:
            raise ValueError(
                f"X has {X.shape[1]} columns; expected {self.d}."
            )
        rotated = X @ self.rotation_
        codes = encode_signs(rotated)
        return codes[0] if squeezed else codes

    def search(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        k: int = 10,
        *,
        return_distances: bool = False,
    ):
        """Top-k Hamming search of ``query`` against an encoded corpus.

        Parameters
        ----------
        query : np.ndarray, shape (d,) or (m, d)
            Raw (un-rotated, un-packed) query vector(s). Will be rotated
            and sign-packed internally.
        codes : np.ndarray, shape (n, d // 8), dtype uint8
            Encoded corpus produced by :meth:`encode`.
        k : int
            Number of neighbours per query.
        return_distances : bool, keyword-only
            If True, also return Hamming distances.

        Returns
        -------
        indices : np.ndarray
            ``(k,)`` for a single query, ``(m, k)`` for a batch.
            Sorted ascending by Hamming distance, ties broken stably.
        distances : np.ndarray
            Same leading shape as ``indices``, dtype ``int64``.
            Only returned if ``return_distances=True``.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        query = np.asarray(query, dtype=self.dtype)
        squeezed = False
        if query.ndim == 1:
            query = query[None, :]
            squeezed = True
        elif query.ndim != 2:
            raise ValueError(f"query must be 1-D or 2-D, got ndim={query.ndim}")
        if query.shape[1] != self.d:
            raise ValueError(
                f"query has {query.shape[1]} columns; expected {self.d}."
            )
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        if codes.ndim != 2 or codes.shape[1] != self.d // 8:
            raise ValueError(
                f"codes shape {codes.shape} incompatible with d={self.d} "
                f"(expected (n, {self.d // 8}))."
            )

        rotated = query @ self.rotation_
        q_codes = encode_signs(rotated)  # (m, d//8)
        m = q_codes.shape[0]
        n = codes.shape[0]
        k_eff = min(k, n)

        out_idx = np.empty((m, k_eff), dtype=np.intp)
        out_dist = np.empty((m, k_eff), dtype=np.int64)

        # Per-query loop — fine for v0.1.0 (O(m·n·B) work either way, and
        # the SIMD popcount kernel that would justify full vectorisation
        # is explicitly post-v0.1.0).
        for i in range(m):
            dists = hamming_distances(codes, q_codes[i])
            if k_eff == n:
                order = np.argsort(dists, kind="stable")
            else:
                part = np.argpartition(dists, k_eff)[:k_eff]
                order = part[np.argsort(dists[part], kind="stable")]
            out_idx[i] = order
            out_dist[i] = dists[order]

        if squeezed:
            out_idx = out_idx[0]
            out_dist = out_dist[0]
        if return_distances:
            return out_idx, out_dist
        return out_idx
