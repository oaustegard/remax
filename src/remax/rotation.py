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

__all__ = ["haar_rotation"]


def haar_rotation(d: int, seed: int | None = None) -> np.ndarray:
    """Generate a Haar-distributed random orthogonal matrix.

    Parameters
    ----------
    d : int
        Matrix dimension. Must be a positive integer.
    seed : int | None, default=None
        RNG seed. Same seed → same matrix on the same machine.

    Returns
    -------
    R : np.ndarray, shape (d, d), dtype float64
        Orthogonal rotation matrix. ``R @ R.T`` is the identity to within
        floating-point round-off.

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
    return Q
