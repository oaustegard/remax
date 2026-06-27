"""remax.stacked — k-stack SimHash with rank-correct precision ladder.

A single SimHash signature (``remax.core.SignBitQuantizer``) is a noisy
estimator of cosine similarity: across ``d`` independent random
directions, the empirical disagreement rate ``h / d`` concentrates around
``θ / π`` (Charikar 2002, Goemans–Williamson 1995) with variance
``(θ/π)(1-θ/π) / d``.

Stacking ``k`` independent SimHash signatures and pooling their disagreement
counts gives a single estimator over ``k · d`` bits with variance
``(θ/π)(1-θ/π) / (k · d)``. Holding ``d`` fixed, this is the **rank-correct
precision ladder**: every step shrinks variance by ``1/k`` while every bit
remains a Charikar-honest sign bit. Unlike Lloyd-Max scalar quantization,
there is no broken middle — adding precision never reorders neighbours
non-monotonically. (See `remex <https://github.com/oaustegard/remex>`_
for the Lloyd-Max contrast and the May 2026 1-bit-beats-2-bit blog post:
https://muninn.austegard.com/blog/one-bit-beats-two.html)

Storage layout
--------------
Codes are stored flat as ``(n, k * d // 8)`` ``uint8``. The ``k`` per-row
signatures sit contiguously within each row, so a Hamming scan against a
query is a single XOR + popcount-LUT sum over the full ``k · d`` bits —
identical code path to the 1-bit case, just over wider rows. The
alternative ``(k, n, d // 8)`` stacked layout was considered (it would
let very large ``n`` keep individual stacks resident in cache), but at
v0.1.0 scales (``n ≲ 100k``) row-contiguous wins on cache locality during
the per-query scan and, more importantly, lets us reuse
``packing.hamming_distances`` without modification.

Composition with remex
----------------------
A future deployment can pair ``remax.StackedSignBitQuantizer`` (rank-correct
scoring) with ``remex.IVFCoarseIndex`` (sublinear cell routing). v0.1.0
ships the building block, not the integration.
"""

from __future__ import annotations

import numpy as np

from .packing import POPCOUNT_LUT, hamming_distances, stable_top_k
from .rotation import haar_rotation

__all__ = ["StackedSignBitQuantizer"]


class StackedSignBitQuantizer:
    """k-stack cosine LSH quantizer (stacked SimHash).

    Holds ``k`` independent Haar rotations. Encoding each input row produces
    ``k`` packed sign-bit signatures concatenated into a single
    ``(k * d // 8)``-byte code. Hamming distance over these codes is — in
    expectation — a monotone function of the angle between the original
    vectors, with variance shrinking as ``1 / k``.

    Parameters
    ----------
    d : int
        Input dimension. Must be a positive integer divisible by 8 (codes
        are bit-packed bytewise).
    k : int
        Number of stacked signatures. Must be a positive integer. ``k=1``
        is equivalent in semantics to :class:`~remax.SignBitQuantizer`,
        but seeded through a :class:`numpy.random.SeedSequence`, so codes
        are not byte-identical to ``SignBitQuantizer(d, seed=master)``.
    seed : int | None, default=None
        Master RNG seed. Per-stack rotations are derived via
        :class:`numpy.random.SeedSequence` so the ``k`` rotations are
        statistically independent while the full ensemble is reproducible
        from ``(d, k, seed)``.
    dtype : numpy dtype, default=np.float32
        Working precision for the rotations and the encode matmul.
        See :class:`~remax.SignBitQuantizer` for the rationale; the
        intermediate (``k`` × ``n`` × ``d``) array dominates memory at
        large ``k``, so the f32 default also halves stacked-encode peak
        RSS. Pass ``np.float64`` for bit-exact compatibility with corpora
        encoded before this default changed.

    Attributes
    ----------
    d : int
        Input dimension.
    k : int
        Number of stacked signatures.
    seed : int | None
        Master seed.
    dtype : numpy dtype
        Working precision (matches ``rotations_.dtype``).
    n_bits : int
        Total bits per code, ``k * d``.
    rotations_ : np.ndarray, shape (k, d, d)
        Stack of ``k`` independent Haar rotation matrices.

    Examples
    --------
    >>> import numpy as np
    >>> from remax import StackedSignBitQuantizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((1000, 768))
    >>> q = StackedSignBitQuantizer(d=768, k=4, seed=42)
    >>> codes = q.encode(X)            # (1000, 384) uint8 — 4 * 768 / 8
    >>> top = q.search(X[0], codes, k=10)
    >>> top[0] == 0
    True

    References
    ----------
    Charikar, M. (2002). "Similarity estimation techniques from rounding
    algorithms." *STOC '02*.
    Goemans, M. & Williamson, D. (1995). "Improved approximation algorithms
    for maximum cut and satisfiability problems using semidefinite
    programming." *J. ACM*.
    """

    def __init__(
        self,
        d: int,
        k: int,
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
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        self.d: int = int(d)
        self.k: int = int(k)
        self.seed: int | None = seed
        self.dtype: np.dtype = np.dtype(dtype)
        self.n_bits: int = self.k * self.d

        # Spawn k independent uint32 seeds from the master via SeedSequence.
        # This is numpy's textbook approach for reproducible parallel streams
        # (https://numpy.org/doc/stable/reference/random/parallel.html) and
        # guarantees independence of the per-stack rotations.
        ss = np.random.SeedSequence(seed)
        child_states = ss.generate_state(self.k, dtype=np.uint32)

        rotations = np.empty((self.k, self.d, self.d), dtype=self.dtype)
        for j in range(self.k):
            rotations[j] = haar_rotation(
                self.d, seed=int(child_states[j]), dtype=self.dtype
            )
        self.rotations_: np.ndarray = rotations

        # Pre-flatten the k rotations into a single (d, k * d) projection
        # matrix so encode() can apply all stacks with one BLAS matmul
        # (X @ self._rotation_matrix) instead of an einsum followed by a
        # transpose-copy of a (k, n, d) intermediate. Stacking the rotations
        # side by side in rotation order, combined with d % 8 == 0, means the
        # packed bits already land in the row-contiguous (n, k * d // 8)
        # layout — no rearrange needed. Output is bit-identical to the einsum
        # path; see encode().
        self._rotation_matrix: np.ndarray = np.ascontiguousarray(
            rotations.transpose(1, 0, 2).reshape(self.d, self.k * self.d)
        )

    # ------------------------------------------------------------------ #
    # sklearn-style API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray | None = None) -> "StackedSignBitQuantizer":
        """No-op fit, retained for API symmetry with sklearn-style transformers.

        The ``k`` rotations are fully determined by ``(d, k, seed)`` and were
        established in ``__init__``. If ``X`` is given, its column count is
        validated against ``self.d`` and a ``ValueError`` is raised on
        mismatch.
        """
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != self.d:
                raise ValueError(
                    f"X has shape {X.shape}; expected (n, {self.d})."
                )
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode ``(n, d)`` float input into ``(n, k * d // 8)`` uint8 codes.

        Accepts a 1-D vector ``(d,)`` as a single-row batch; in that case
        the returned shape is ``(k * d // 8,)``.

        Layout: within each row, the ``k`` packed signatures sit contiguously,
        in the order of ``self.rotations_``. So the first ``d // 8`` bytes
        of row ``i`` are the SimHash code under rotation 0, the next
        ``d // 8`` are under rotation 1, and so on.
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

        # Apply all k rotations with a single BLAS matmul against the
        # pre-flattened (d, k * d) projection matrix built in __init__.
        # rotated[n, j*d + e] = sum_d X[n, d] · rotations_[j, d, e], i.e. the
        # k per-stack projections concatenated along the column axis in
        # rotation order.
        rotated = X @ self._rotation_matrix  # (n, k * d)

        # Sign-pack the full row in one go. packbits is big-endian within each
        # byte (matching encode_signs in packing.py); because each stack spans
        # exactly d bits and d % 8 == 0, signature boundaries fall on byte
        # boundaries, so the result is already the row-contiguous
        # (n, k * d // 8) layout described in the class docstring — no
        # transpose-copy of a (k, n, d) intermediate.
        codes = np.packbits(rotated > 0, axis=1)  # (n, k * d // 8) uint8

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
            through all ``k`` stacks and sign-packed internally.
        codes : np.ndarray, shape (n, k * d // 8), dtype uint8
            Encoded corpus produced by :meth:`encode`.
        k : int
            Number of neighbours per query. (Note: distinct from
            ``self.k``, the stack count. The two share a name only because
            both are conventional.)
        return_distances : bool, keyword-only
            If True, also return Hamming distances.

        Returns
        -------
        indices : np.ndarray
            ``(k,)`` for a single query, ``(m, k)`` for a batch.
            Sorted ascending by Hamming distance, ties broken stably.
        distances : np.ndarray
            Same leading shape as ``indices``, dtype ``int64``,
            in ``[0, k * d]``.
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
        expected_bytes = self.k * (self.d // 8)
        if codes.ndim != 2 or codes.shape[1] != expected_bytes:
            raise ValueError(
                f"codes shape {codes.shape} incompatible with "
                f"(d={self.d}, k={self.k}); expected (n, {expected_bytes})."
            )

        q_codes = self.encode(query)  # (m, k * d // 8)
        m = q_codes.shape[0]
        n = codes.shape[0]
        k_eff = min(k, n)

        out_idx = np.empty((m, k_eff), dtype=np.intp)
        out_dist = np.empty((m, k_eff), dtype=np.int64)

        # Per-query scan. The XOR + LUT sum is over k * d bits per row,
        # but the operation is the same shape as the 1-bit case — only
        # the column count changes. SIMD popcount is post-v0.1.0.
        for i in range(m):
            dists = hamming_distances(codes, q_codes[i])
            order = stable_top_k(dists, k_eff)
            out_idx[i] = order
            out_dist[i] = dists[order]

        if squeezed:
            out_idx = out_idx[0]
            out_dist = out_dist[0]
        if return_distances:
            return out_idx, out_dist
        return out_idx
