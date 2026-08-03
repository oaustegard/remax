"""remax.rotation — orthogonal rotation matrices for SimHash projection.

The cosine-LSH guarantee from Charikar (2002) and Goemans–Williamson (1995)
relies on projecting onto random isotropic directions. Two constructions
are available:

``haar_rotation`` (default)
    The textbook QR construction:

      1. ``A ~ N(0, I_{d×d})`` via ``np.random.default_rng(seed).standard_normal``.
      2. ``Q, R = np.linalg.qr(A)`` (LAPACK ``dgeqrf`` underneath).
      3. Mezzadri (2007) sign correction so ``Q`` is uniformly distributed on
         the orthogonal group ``O(d)`` rather than on a fundamental domain of it.

    LAPACK QR is bit-deterministic on a single machine but can drift across
    BLAS builds; that's an acceptable v0.1.0 limitation. (See remex's explicit
    Householder implementation in ``remex/rotation.py`` for the cross-BLAS
    deterministic pattern, kept out of scope here.)

``rht_rotation``
    A randomized Hadamard transform materialized as a dense ``(d, d)``
    matrix. ``O(d² log d)`` to build instead of ``O(d³)``: 1.5–1.8× faster
    than the QR path measured over the ``k ∈ {1, 2, 4, 8}`` ladder at
    ``d ∈ {768, 1024, 3072}``. Materializing keeps the encode path a single
    BLAS matmul, which matters — see the note in :func:`rht_rotation` about
    the operator form, which is *slower*.

Why the RHT needs care here (issue #59)
---------------------------------------
An RHT is **not** Haar-distributed. It carries ``O(d log d)`` bits of
randomness against Haar's ``O(d²)``, and its projection directions are
structured rather than independent — so the Charikar collision bound, which
is a distributional statement, has to be re-measured rather than inherited
from remex's use of the same substitution for a *reconstruction* codec
(remex#71), where only isotropy mattered.

It was measured (``bench/rotation_lsh_fidelity.py``), and the substitution
holds **only with at least two mixing rounds**. remex's construction uses a
single round when ``d`` is a power of two; at one round the ``k`` stacked
rotations stop being independent on anisotropic input, which is precisely
the assumption ``StackedSignBitQuantizer`` rests on:

======================  ==========  ==========  ===========
d=1024, k=8, θ/π=0.5    rounds=1    rounds=2    haar
======================  ==========  ==========  ===========
bias vs θ/π             +0.0026     −0.0005     +0.0001
cross-stack mean |corr| 0.117       0.043       0.039
Var(pooled)·k/Var(one)  1.82        1.03        0.94
======================  ==========  ==========  ===========

A ``Var(pooled)·k/Var(one)`` of 1.82 means a ``k=8`` stack delivered the
variance of ``k≈4.4`` — half the precision ladder silently lost, and the
bias does not shrink as ``k`` grows because every stack shares the same
structural defect. Hence :func:`rht_rotation` floors ``rounds`` at 2. Two
rounds cost ~0.04 s at ``d=768`` and ~1.6 s at ``d=3072``, so the guard is
close to free.

With that floor, end-to-end recall@10 against exact-cosine ground truth is
statistically indistinguishable from Haar on three real corpora
(jina-v5-nano/SciFact d=768, Gemini d=1024 and d=3072) across
``k ∈ {1, 2, 4, 8}`` and 3 seeds — pooled Δ ``+0.0034``, against a per-cell
seed spread of ±0.002–0.013. See ``bench/results/ROTATION_LSH.md``.

``"haar"`` remains the default. The two constructions produce different codes
from the same ``(d, k, seed)``, and nothing in a persisted index records which
was used, so flipping the default would silently invalidate stored corpora.
That is a format decision, not a performance one.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["haar_rotation", "rht_rotation", "build_rotation", "ROTATIONS"]


def haar_rotation(
    d: int,
    seed: int | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Generate a Haar-distributed random orthogonal matrix.

    Parameters
    ----------
    d : int
        Matrix dimension. Must be a positive integer.
    seed : int | None, default=None
        RNG seed. Same seed → same matrix on the same machine.
    dtype : numpy dtype, default=np.float32
        Output dtype. The QR factorisation runs in float64 for numerical
        stability (Mezzadri sign correction depends on the sign of
        ``diag(R)``, which is sharp in f64 even for nearly-singular
        diagonals); the result is then cast. f32 is the default because
        SimHash only consumes the *sign* of ``X @ R``, so f64 precision in
        the rotation matrix is wasted bandwidth — see ``packing.encode_signs``.

    Returns
    -------
    R : np.ndarray, shape (d, d), dtype matches ``dtype``
        Orthogonal rotation matrix. ``R @ R.T`` is the identity to within
        floating-point round-off (looser at f32: ``atol≈1e-5`` instead of
        ``≈1e-12``).

    References
    ----------
    Mezzadri, F. (2007). "How to generate random matrices from the
    classical compact groups." *Notices of the AMS*, 54(5), 592–604.
    """
    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    # Mezzadri sign correction: multiplying each column of Q by the sign of
    # the corresponding diagonal of R yields a Haar-uniform Q. Scaling
    # columns by ±1 preserves orthogonality, so the test ``Q @ Q.T ≈ I``
    # still holds.
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs  # broadcasts over columns
    return Q.astype(dtype, copy=False)


# ---------------------------------------------------------------------- #
# Randomized Hadamard transform
# ---------------------------------------------------------------------- #
#: Minimum mixing rounds for :func:`rht_rotation`. One round leaves the ``k``
#: rotations of a stack correlated on anisotropic input — see the module
#: docstring for the measurement. Do not lower this without re-running
#: ``bench/rotation_lsh_fidelity.py``.
_MIN_RHT_ROUNDS = 2


def _largest_pow2_divisor(d: int) -> int:
    """Largest power of two dividing ``d``. The FWHT block size."""
    b = 1
    while d % (b * 2) == 0:
        b *= 2
    return b


def _fwht_inplace(y: np.ndarray) -> None:
    """Unnormalized fast Walsh-Hadamard transform along the last axis, in place.

    ``y`` must be ``(..., B)`` with ``B`` a power of two. The caller applies
    the ``1/sqrt(B)`` normalisation that makes the transform orthogonal.
    """
    B = y.shape[-1]
    h = 1
    while h < B:
        y2 = y.reshape(-1, B // (2 * h), 2, h)
        a = y2[:, :, 0, :].copy()
        b = y2[:, :, 1, :]
        y2[:, :, 0, :] = a + b
        y2[:, :, 1, :] = a - b
        h *= 2


def rht_rotation(
    d: int,
    seed: int | None = None,
    dtype: np.dtype | type = np.float32,
    *,
    rounds: int | None = None,
) -> np.ndarray:
    """Randomized Hadamard rotation, materialized as a dense ``(d, d)`` matrix.

    Same contract as :func:`haar_rotation` — a deterministic-from-seed
    orthogonal matrix — but built in ``O(d² log d)`` instead of ``O(d³)``.
    It is *not* Haar-distributed; see the module docstring for what was
    measured to justify using it for SimHash anyway, and for why ``rounds``
    is floored at 2.

    Construction, for any even ``d`` rather than powers of two only: rounds of
    (permute → sign flip → block-diagonal FWHT) with block size the largest
    power of two dividing ``d``. Padding ``d`` up to a power of two would
    change the code width, so it is not an option. The transform is
    materialized by applying it to the identity in one batched pass, which
    keeps the encode path a single BLAS matmul — an operator form that
    applies the FWHT per batch was benchmarked and is 7–20× *slower* than
    ``sgemm`` under NumPy, despite ~100× fewer flops.

    Parameters
    ----------
    d : int
        Matrix dimension. Must be a positive even integer (the FWHT needs a
        block size of at least 2). ``remax`` quantizers require
        ``d % 8 == 0``, which satisfies this.
    seed : int | None, default=None
        RNG seed. Same seed → same matrix, on any BLAS (unlike
        :func:`haar_rotation`, this path runs no LAPACK).
    dtype : numpy dtype, default=np.float32
        Output dtype. Unlike :func:`haar_rotation`, which factorises in f64
        and casts, this builds directly in the requested precision (promoted
        to at least f32): the FWHT is a sum of ``±`` terms with no
        cancellation to guard against, so f64 working precision would buy an
        f32 caller nothing. Orthogonality lands at machine epsilon for the
        output dtype — and is *exact* when the FWHT block size is an even
        power of two, since ``1/sqrt(B)`` is then itself a power of two.
    rounds : int | None, keyword-only, default=None
        Mixing rounds. ``None`` selects ``max(2, ceil(log d / log B))``,
        enough passes for every coordinate to reach every other. Values
        below 2 are rejected: see the module docstring.

    Returns
    -------
    R : np.ndarray, shape (d, d), dtype matches ``dtype``
        Orthogonal rotation matrix.

    Raises
    ------
    ValueError
        If ``d`` is not a positive even integer, or ``rounds < 2``.

    References
    ----------
    Ailon, N. & Chazelle, B. (2009). "The fast Johnson-Lindenstrauss
    transform and approximate nearest neighbors." *SIAM J. Comput.*
    """
    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    B = _largest_pow2_divisor(int(d))
    if B < 2:
        raise ValueError(
            f"d={d} is odd; the randomized Hadamard construction needs an "
            "even dimension. Use haar_rotation instead."
        )
    d = int(d)
    if rounds is None:
        rounds = max(_MIN_RHT_ROUNDS, math.ceil(math.log(d) / math.log(B)))
    elif rounds < _MIN_RHT_ROUNDS:
        raise ValueError(
            f"rounds={rounds} is below the minimum of {_MIN_RHT_ROUNDS}. "
            "A single mixing round leaves the k rotations of a stack "
            "correlated on anisotropic input, costing roughly half the "
            "1/k variance reduction the precision ladder is built on "
            "(remax#59); see remax.rotation's module docstring."
        )

    # f32 floor: integer or f16 dtypes cannot represent the intermediates.
    work = np.promote_types(np.dtype(dtype), np.float32)
    rng = np.random.default_rng(seed)
    # Apply the transform to the identity, one batched pass over all d rows.
    Y = np.eye(d, dtype=work)
    scale = work.type(1.0 / math.sqrt(B))
    plus_minus = np.array([-1.0, 1.0], dtype=work)
    for _ in range(rounds):
        perm = rng.permutation(d)
        sign = rng.choice(plus_minus, size=d)
        Y = Y[:, perm] * sign
        Y = np.ascontiguousarray(Y.reshape(d, d // B, B))
        _fwht_inplace(Y)
        Y = Y.reshape(d, d) * scale
    return Y.astype(dtype, copy=False)


#: Rotation constructions selectable by name on the quantizer classes.
ROTATIONS = {"haar": haar_rotation, "rht": rht_rotation}


def build_rotation(
    kind: str,
    d: int,
    seed: int | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Dispatch to the rotation construction named by ``kind``.

    Parameters
    ----------
    kind : {"haar", "rht"}
        ``"haar"`` for the Haar-distributed QR construction, ``"rht"`` for
        the randomized Hadamard transform.
    d, seed, dtype
        Forwarded to the selected constructor.

    Raises
    ------
    ValueError
        If ``kind`` is not a known rotation name.
    """
    try:
        fn = ROTATIONS[kind]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown rotation {kind!r}; expected one of "
            f"{sorted(ROTATIONS)}."
        ) from None
    return fn(d, seed=seed, dtype=dtype)
