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
from .packing import (
    as_codes,
    asymmetric_scores,
    asymmetric_tables,
    encode_signs,
    hamming_distances,
    hamming_topk_batch,
    scores_from_table,
    stable_top_k,
)
from .rotation import ROTATIONS, build_rotation, haar_rotation, rht_rotation

#: Byte budget for the batched asymmetric table block. 64 MB holds m=256
#: queries at d=2048; larger batches are processed in groups of that size.
_ASYM_TABLE_BUDGET = 64 << 20

__all__ = [
    "SignBitQuantizer",
    "asymmetric_scores",
    "haar_rotation",
    "rht_rotation",
    "encode_signs",
    "hamming_distances",
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
    rotation : {"haar", "rht"}, default="haar"
        Rotation construction. ``"haar"`` is the Haar-distributed QR path.
        ``"rht"`` is a randomized Hadamard transform — cheaper to build
        (``O(d² log d)`` vs ``O(d³)``) and measured equivalent for retrieval
        (remax#59 — see :mod:`remax.rotation`). Codes are **not**
        interchangeable between the two.

    Attributes
    ----------
    d : int
        Input dimension.
    seed : int | None
        RNG seed used for the rotation.
    dtype : numpy dtype
        Working precision (matches ``rotation_.dtype``).
    rotation : str
        The rotation construction in use.
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
        rotation: str = "haar",
    ):
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        if d % 8 != 0:
            raise ValueError(
                f"d must be divisible by 8 (got d={d}); remax codes are "
                "bit-packed into uint8 bytes."
            )
        if rotation not in ROTATIONS:
            raise ValueError(
                f"unknown rotation {rotation!r}; expected one of "
                f"{sorted(ROTATIONS)}."
            )
        self.d: int = int(d)
        self.seed: int | None = seed
        self.dtype: np.dtype = np.dtype(dtype)
        self.rotation: str = rotation
        self.rotation_: np.ndarray = build_rotation(
            rotation, self.d, seed=seed, dtype=self.dtype
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
        threads: int | str | None = None,
        block: int | None = None,
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
        threads : int | "auto" | None, keyword-only
            Threads for the scan. ``None`` uses the process-wide default
            (:func:`remax.packing.get_default_threads`, 1 unless changed), so
            the default behaviour is unchanged. The ctypes kernel releases the
            GIL, so this is real parallelism; results are bit-identical at any
            thread count because the decomposition is a row partition.
        block : int | None, keyword-only
            Corpus rows held per pass. ``None`` blocks automatically, for two
            different reasons: a batch (``m >= 2``) so the corpus is read once
            per block for all queries instead of once per query, and a single
            large query so its distances are filtered against a running
            threshold while still in cache rather than written out in full and
            selected over. ``block=len(codes)`` forces the old per-query full
            pass. Output is bit-identical either way.

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
        # Shape first, then as_codes. Reversing these would emit a
        # "copying N MB" warning for a codes array that is about to be
        # rejected as the wrong width anyway — a slice like codes[:, :-1] is
        # both non-contiguous and wrong, and the useful message is the second.
        _codes = np.asarray(codes)
        if _codes.ndim != 2 or _codes.shape[1] != self.d // 8:
            raise ValueError(
                f"codes shape {_codes.shape} incompatible with d={self.d} "
                f"(expected (n, {self.d // 8}))."
            )
        # Validated once here, and once only: as_codes is the single site in
        # the library allowed to copy a code matrix, and it warns when it
        # does. hamming_distances re-checks (cheaply — no copy is possible on
        # an already-good array) because it is public in its own right.
        codes = as_codes(_codes)

        rotated = query @ self.rotation_
        q_codes = encode_signs(rotated)  # (m, d//8)
        n = codes.shape[0]
        k_eff = min(k, n)

        # The m-loop, the row blocking and the scan threading all live in
        # packing.hamming_topk_batch — one implementation, one place for the
        # blocked/unblocked equivalence to be checked. int64 distances are the
        # documented public dtype; the kernel works in int32.
        out_idx, out_dist32 = hamming_topk_batch(
            codes, q_codes, k_eff, threads=threads, block=block
        )
        out_dist = out_dist32.astype(np.int64)

        if squeezed:
            out_idx = out_idx[0]
            out_dist = out_dist[0]
        if return_distances:
            return out_idx, out_dist
        return out_idx

    def search_asymmetric(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        k: int = 10,
        *,
        return_scores: bool = False,
    ):
        """Top-k by float-query-against-sign-bit dot product.

        Same stored index as :meth:`search` -- the corpus is still one bit per
        dimension. The only difference is that the query is NOT binarized: it
        is rotated and kept in float, then scored against the +/-1 values the
        code bits stand for.

        That asymmetry is free. The query is a single vector per search and
        occupies no index storage, so discarding its precision to make the
        comparison symmetric buys nothing. Measured on LFM2.5/SciFact it is
        worth +0.019 nDCG@10 at 128 B/vector and +0.084 at 16 B -- the harder
        the compression, the more the query precision is carrying
        (bench/asymmetric_lfm25.py). Exa's web-scale index makes the same
        choice.

        Trade-off vs :meth:`search`: scores are float rather than integer
        Hamming distances, so this cannot use the SIMD popcount kernel and
        costs a gather-and-sum per byte instead. Prefer :meth:`search` when
        throughput dominates and recall is adequate; prefer this when recall
        per stored byte matters, which is the usual reason to be at 1 bit.

        Parameters
        ----------
        query : np.ndarray, shape (d,) or (m, d)
            Raw (un-rotated) query vector(s).
        codes : np.ndarray, shape (n, self.d // 8), dtype uint8
            Encoded corpus from :meth:`encode` -- unchanged, no re-encoding.
        k : int
            Number of neighbours per query.
        return_scores : bool, keyword-only
            If True, also return the dot products of the top-k.

        Returns
        -------
        indices : np.ndarray
            ``(k,)`` for a single query, ``(m, k)`` for a batch.
            Sorted DESCENDING by score (higher is nearer), ties broken stably.
        scores : np.ndarray
            Same leading shape, dtype ``float32``. Only if
            ``return_scores=True``.
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
        n = codes.shape[0]
        n_bytes = codes.shape[1]
        m = query.shape[0]
        k_eff = min(k, n)
        out_idx = np.empty((m, k_eff), dtype=np.intp)
        out_sc = np.empty((m, k_eff), dtype=np.float32)

        # Build every query's (B, 256) byte table before the loop, in one
        # batched GEMM, instead of one small GEMM per iteration. The tables
        # depend only on the queries, so rebuilding them inside the loop was
        # pure repetition: 136 us per query at d=2048 on the box in
        # bench/results/QUERY_PATH_SPEED.md. Bit-identical — the arithmetic is
        # the same contraction, just batched (tests/test_asymmetric.py).
        #
        # Queries are grouped so the table block stays bounded: m * B * 256 * 4
        # bytes would be 200 MB for m=1000 at d=2048.
        group = max(1, _ASYM_TABLE_BUDGET // max(1, n_bytes * 256 * 4))
        scratch = np.empty(n, dtype=np.float32)
        for lo in range(0, m, group):
            hi = min(lo + group, m)
            tables, offsets = asymmetric_tables(rotated[lo:hi], n_bytes)
            for j in range(hi - lo):
                scores = scores_from_table(
                    tables[j], codes, float(offsets[j]), out=scratch
                )
                order = stable_top_k(-scores, k_eff)
                out_idx[lo + j] = order
                out_sc[lo + j] = scores[order]

        if squeezed:
            out_idx = out_idx[0]
            out_sc = out_sc[0]
        if return_scores:
            return out_idx, out_sc
        return out_idx
