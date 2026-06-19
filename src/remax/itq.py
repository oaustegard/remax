"""remax.itq — learned-rotation SimHash (ITQ) with the same ladder shape.

:class:`StackedITQQuantizer` is the learned-projection counterpart of
:class:`remax.StackedSignBitQuantizer`. Where the stacked baseline draws ``k``
*Haar-random* rotations, this draws ``k`` *learned* rotations: each is an
Iterative Quantization (ITQ; Gong & Lazebnik 2011) fit that aligns the sign
boundaries to the data so the Hamming code keeps more cosine rank order at the
same bit budget.

The encode/search/storage path is **identical** to the stacked baseline — same
``(n, k·d/8)`` byte layout, same XOR+popcount scan, same ladder semantics. Only
the rotations change (learned, plus a stored training mean), so the
stacked-precision ladder is preserved exactly: ``k`` learned signatures pool the
same way ``k`` random ones do.

Two costs distinguish it from the parameter-free baseline:

* a ``fit(X_train)`` step that learns the ``k`` rotations (one SVD-driven ITQ
  optimisation per stack), and
* a corpus-specific artefact (``k`` ``d×d`` rotations + one ``d``-vector mean)
  that must travel with the index.

Whether that buys enough rank-correctness to justify giving up SimHash's
zero-training robustness is the empirical question of issue #46; the
parameter-free :class:`~remax.StackedSignBitQuantizer` remains the default.

Centring
--------
Plain SimHash and ITQ both place the sign boundary at the origin, so both need
mean-zero inputs (real embeddings are not — SPECTER2 has a dim with mean ≈ 15.5,
cf. ``run_baseline``'s ``center`` flag). This quantiser centres internally:
``fit`` records ``mean_`` from the training rows and ``encode`` subtracts it
from everything. For a transfer fit (train on corpus A, encode corpus B) the
*training* mean travels with the rotation — the learned artefact is "rotation +
its centring", applied verbatim to new data.
"""

from __future__ import annotations

import numpy as np

from .rotation import itq_rotation
from .stacked import StackedSignBitQuantizer

__all__ = ["StackedITQQuantizer"]


class StackedITQQuantizer(StackedSignBitQuantizer):
    """k-stack SimHash whose rotations are learned by ITQ rather than Haar-random.

    Construction only fixes ``(d, k, seed, n_iters, dtype)``; the ``k``
    rotations are **not** materialised until :meth:`fit` is given training
    data. This differs from :class:`~remax.StackedSignBitQuantizer` (whose
    rotations are fully determined at construction) precisely because ITQ
    needs the data — and it is what lets the transfer experiment fit on one
    corpus and encode another.

    Parameters
    ----------
    d : int
        Input dimension. Positive and divisible by 8 (codes are bit-packed
        bytewise), same constraint as the baseline.
    k : int
        Number of stacked learned signatures. ``k=1`` is the single-rotation
        "learned 1-bit" code; ``k=2,4,8`` walk the precision ladder. Total
        bits per code is ``k·d`` — identical to the baseline at the same
        ``k``, so the comparison is at equal bit budget.
    seed : int | None, default=None
        Master seed. Per-stack ITQ initialisations are spawned via
        :class:`numpy.random.SeedSequence`, exactly as the baseline spawns its
        per-stack Haar seeds, so the ``k`` learned rotations start from
        independent random frames and the full fit is reproducible from
        ``(d, k, seed, n_iters, X_train)``.
    n_iters : int, default=50
        ITQ iterations per stack (see :func:`remax.rotation.itq_rotation`).
    dtype : numpy dtype, default=np.float32
        Working precision for the stored rotations and the encode matmul.

    Attributes
    ----------
    rotations_ : np.ndarray | None
        ``(k, d, d)`` learned rotations once fitted, else ``None``.
    mean_ : np.ndarray | None
        ``(d,)`` training mean subtracted at encode time once fitted, else
        ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from remax.itq import StackedITQQuantizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((2000, 64))
    >>> q = StackedITQQuantizer(d=64, k=4, seed=42).fit(X)
    >>> codes = q.encode(X)              # (2000, 32) uint8 — 4 * 64 / 8
    >>> top = q.search(X[0], codes, k=10)
    >>> top[0] == 0
    True
    """

    def __init__(
        self,
        d: int,
        k: int,
        seed: int | None = None,
        *,
        n_iters: int = 50,
        dtype: np.dtype | type = np.float32,
    ):
        # Validate shape/budget args exactly as the baseline does, but DO NOT
        # call super().__init__ — that would eagerly build Haar rotations.
        # ITQ rotations are learned in fit().
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        if d % 8 != 0:
            raise ValueError(
                f"d must be divisible by 8 (got d={d}); remax codes are "
                "bit-packed into uint8 bytes."
            )
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if not isinstance(n_iters, (int, np.integer)) or n_iters <= 0:
            raise ValueError(
                f"n_iters must be a positive integer, got {n_iters!r}"
            )
        self.d: int = int(d)
        self.k: int = int(k)
        self.seed: int | None = seed
        self.n_iters: int = int(n_iters)
        self.dtype: np.dtype = np.dtype(dtype)
        self.n_bits: int = self.k * self.d
        self.rotations_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # fit — the part that actually differs from the baseline
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "StackedITQQuantizer":
        """Learn the ``k`` ITQ rotations and the centring mean from ``X``.

        Records ``mean_ = X.mean(0)`` and learns each of the ``k`` rotations
        on the centred training data ``X - mean_`` from an independent
        SeedSequence-spawned initialisation. Returns ``self`` for chaining.

        ``X`` must be 2-D with ``d`` columns. Unlike the baseline's no-op
        ``fit``, this one is required before :meth:`encode` / :meth:`search`.
        """
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(
                f"X has shape {X.shape}; expected (n, {self.d})."
            )
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Spawn k independent uint32 seeds from the master — same mechanism the
        # baseline uses for its Haar stacks, so "stack independence" means the
        # same thing here: independent random *starts*. (ITQ rotations still
        # optimise a shared objective, so they decorrelate less than Haar
        # draws; that interaction with the ladder is what issue #46 measures.)
        ss = np.random.SeedSequence(self.seed)
        child_states = ss.generate_state(self.k, dtype=np.uint32)

        rotations = np.empty((self.k, self.d, self.d), dtype=self.dtype)
        for j in range(self.k):
            rotations[j] = itq_rotation(
                Xc,
                n_iters=self.n_iters,
                seed=int(child_states[j]),
                dtype=self.dtype,
            )
        self.rotations_ = rotations
        return self

    # ------------------------------------------------------------------ #
    # encode — centre, then defer to the baseline's stacked encode path
    # ------------------------------------------------------------------ #
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Centre by the training mean, then encode via the stacked path.

        Identical output layout to :meth:`StackedSignBitQuantizer.encode`
        (``(n, k·d/8)`` uint8, per-row-contiguous signatures). Raises if the
        quantiser has not been :meth:`fit`.
        """
        if self.rotations_ is None or self.mean_ is None:
            raise RuntimeError(
                "StackedITQQuantizer must be fit(X_train) before encode/search."
            )
        X = np.asarray(X, dtype=self.dtype)
        # Subtract the training mean; broadcasting covers both a (d,) vector
        # and an (n, d) batch. Validation of column count is left to the
        # inherited encode, which raises with the canonical message.
        if X.ndim in (1, 2) and X.shape[-1] == self.d:
            X = X - self.mean_
        return super().encode(X)
