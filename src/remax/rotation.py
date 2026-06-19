"""remax.rotation — orthogonal rotation matrices for SimHash projection.

The cosine-LSH guarantee from Charikar (2002) and Goemans–Williamson (1995)
relies on projecting onto random isotropic directions. A Haar-distributed
orthogonal matrix achieves this in dense form; structured alternatives
(Hadamard, FFHT) belong to a future issue.

For v0.1.0 we use the textbook QR construction:

  1. ``A ~ N(0, I_{d×d})`` via ``np.random.default_rng(seed).standard_normal``.
  2. ``Q, R = np.linalg.qr(A)`` (LAPACK ``dgeqrf`` underneath).
  3. Mezzadri (2007) sign correction so ``Q`` is uniformly distributed on
     the orthogonal group ``O(d)`` rather than on a fundamental domain of it.

LAPACK QR is bit-deterministic on a single machine but can drift across
BLAS builds; that's an acceptable v0.1.0 limitation. (See remex's explicit
Householder implementation in ``remex/rotation.py`` for the cross-BLAS
deterministic pattern, kept out of scope here.)
"""

from __future__ import annotations

import numpy as np

__all__ = ["haar_rotation", "itq_rotation"]


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


def itq_rotation(
    X: np.ndarray,
    n_iters: int = 50,
    seed: int | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Learn an orthogonal rotation minimising sign-bit quantisation error (ITQ).

    Iterative Quantization (Gong & Lazebnik 2011) finds an orthogonal ``R``
    that minimises the quantisation loss of the sign code,

        ``min_{R: RᵀR=I, B∈{±1}} ‖B − X R‖²_F`` ,

    by alternating two closed-form steps from a random orthogonal start:

      1. **Fix R, solve B.** The minimiser is ``B = sign(X R)`` (each entry
         independently picks the nearer of ``±1`` to its projection).
      2. **Fix B, solve R.** This is the orthogonal Procrustes problem
         ``max_R tr(Rᵀ Xᵀ B)``. With the thin SVD ``Xᵀ B = U Σ Vᵀ`` the
         optimum is ``R = U Vᵀ``.

    Each step is non-increasing in the loss, so the loss converges (to a
    local optimum — ITQ is non-convex, hence the random start and the
    ``seed``). Unlike a Haar rotation, the result is **data-fitted**: the
    sign boundaries are aligned to the principal axes of ``X`` so that
    ``sign(X R)`` loses less rank information at the same bit budget. The
    trade is a stored ``d×d`` artefact and a corpus-specific fit — see
    issue #46 for the transfer/robustness characterisation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Training rows. **Should be centred** (mean-subtracted) by the caller;
        ITQ's sign boundary sits at the origin exactly as plain SimHash's
        does, so an off-origin mean degrades the learned code the same way it
        degrades SimHash. :class:`remax.itq.StackedITQQuantizer` handles the
        centring; call this primitive on already-centred data. ``d`` need not
        be divisible by 8 here (the bit-packing constraint lives at the
        quantiser layer); the returned rotation is square ``d×d``.
    n_iters : int, default=50
        Number of alternating ITQ iterations. 50 is the value used in the
        original paper and is well past the convergence knee on embedding
        data; the loss is monotone non-increasing so over-iterating is safe,
        just wasted work.
    seed : int | None, default=None
        RNG seed for the random orthogonal initialisation. Same seed + same
        ``X`` → same rotation (the iteration is deterministic given the
        start). Different seeds explore different local optima — this is what
        makes a *stack* of independently-seeded ITQ rotations more than a
        single rotation repeated.
    dtype : numpy dtype, default=np.float32
        Output dtype. The iteration runs in float64 for SVD stability (the
        Procrustes step depends on singular vectors, sharp in f64); the
        result is cast on return, matching :func:`haar_rotation`. SimHash
        only consumes ``sign(X R)`` so f32 storage is lossless for the code.

    Returns
    -------
    R : np.ndarray, shape (d, d), dtype matches ``dtype``
        Orthogonal rotation. ``R @ R.T`` is the identity to floating-point
        round-off, so it drops straight into the SimHash encode path.

    References
    ----------
    Gong, Y. & Lazebnik, S. (2011). "Iterative Quantization: A Procrustean
    Approach to Learning Binary Codes." *CVPR 2011*.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d), got shape {X.shape}")
    if not isinstance(n_iters, (int, np.integer)) or n_iters <= 0:
        raise ValueError(f"n_iters must be a positive integer, got {n_iters!r}")
    d = X.shape[1]
    # Random orthogonal start. Reuse the Haar generator at f64 for a
    # well-conditioned initial frame.
    R = haar_rotation(d, seed=seed, dtype=np.float64)
    for _ in range(int(n_iters)):
        Z = X @ R                       # (n, d) projections
        B = np.where(Z >= 0.0, 1.0, -1.0)  # nearest ±1 code (sign, 0→+1)
        # Orthogonal Procrustes: max_R tr(Rᵀ Xᵀ B) → R = U Vᵀ for XᵀB=UΣVᵀ.
        U, _s, Vt = np.linalg.svd(X.T @ B, full_matrices=False)
        R = U @ Vt
    return R.astype(dtype, copy=False)
