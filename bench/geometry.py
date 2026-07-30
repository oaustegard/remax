"""Embedding-geometry diagnostics — the cheap predictor of which codec will win.

The bench history in this repo (SPECTER2 case study, NEMOTRON_MASTER, remex's
own README) converges on one point: bit-depth ordering is a property of the
*embedding geometry*, not of the codec. Anisotropic, tightly-clustered encoders
push the Lloyd-Max boundaries outside the region where the mass actually lives,
and 1-bit sign hashing wins the low-bit regime; isotropic encoders invert that.

So before running any retrieval, measure the geometry and state a prediction.
Getting the prediction on the record first is what makes the retrieval numbers
worth something — otherwise the story is written after the fact.

Reference points measured elsewhere in this repo:
  SPECTER2 (anisotropic, 1-bit wins low bits): norm std/mean 0.006,
    post-rotation per-coord sigma ratio 0.389, KS rejects on 20/20 coords.
  Jina v5 nano (isotropic, remex wins): sigma ratio ~1.0, 4-bit near-lossless.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from remax.rotation import haar_rotation


def _participation_ratio(eigvals: np.ndarray) -> float:
    """Effective dimensionality: (sum l)^2 / sum(l^2). Equals d when isotropic."""
    s1 = float(eigvals.sum())
    s2 = float((eigvals**2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def describe(X: np.ndarray, *, seed: int = 42, n_coords: int = 20) -> dict:
    """Geometry report for a raw (pre-normalization) embedding matrix."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    norms = np.linalg.norm(X, axis=1)

    Xu = X / np.maximum(norms, 1e-12)[:, None]
    mu = Xu.mean(axis=0)

    # Anisotropy of the unit-sphere cloud.
    cov = np.cov(Xu, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1].clip(min=0)
    pr = _participation_ratio(eigvals)

    # remex path: Haar-rotate the unit vectors, then compare the per-coordinate
    # spread against the 1/sqrt(d) the Lloyd-Max codebook is built for.
    R = haar_rotation(d, seed=seed).astype(np.float64)
    Xr = Xu @ R.T
    per_coord_sigma = Xr.std(axis=0)
    expected_sigma = 1.0 / np.sqrt(d)
    sigma_ratio = float(per_coord_sigma.mean() / expected_sigma)

    # Gaussianity of the rotated marginals, on a sample of coordinates.
    rng = np.random.default_rng(seed)
    probe = rng.choice(d, size=min(n_coords, d), replace=False)
    ks_reject = 0
    for c in probe:
        col = Xr[:, c]
        s = col.std()
        if s <= 0:
            ks_reject += 1
            continue
        if stats.kstest((col - col.mean()) / s, "norm").pvalue < 0.01:
            ks_reject += 1
    kurt = float(stats.kurtosis(Xr[:, probe], axis=0).mean())

    # remax path: centering is what buys the 1-bit codes their signal. If the
    # cloud sits far off the origin, the uncentered sign bits are near-constant.
    mean_norm = float(np.linalg.norm(mu))
    off_center = mean_norm / float(np.sqrt((Xu**2).sum(axis=1).mean()))

    return {
        "n": int(n),
        "d": int(d),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_cv": float(norms.std() / norms.mean()),
        "mean_vector_norm": mean_norm,
        "off_center_ratio": off_center,
        "eig_top1_frac": float(eigvals[0] / eigvals.sum()),
        "eig_top10_frac": float(eigvals[:10].sum() / eigvals.sum()),
        "participation_ratio": pr,
        "participation_frac": pr / d,
        "post_rot_sigma_mean": float(per_coord_sigma.mean()),
        "post_rot_sigma_expected": float(expected_sigma),
        "post_rot_sigma_ratio": sigma_ratio,
        "post_rot_kurtosis": kurt,
        "ks_reject_frac": ks_reject / len(probe),
    }


def predict(g: dict) -> dict:
    """Turn the geometry into a falsifiable prediction about codec ordering.

    Thresholds come from this repo's measured reference points, not theory:
    remex's README calls out sigma_ratio well under 1 as the Lloyd-Max failure
    mode, and the SPECTER2 case study measured 0.389 on the encoder where 1-bit
    beat 2- and 3-bit.
    """
    sr = g["post_rot_sigma_ratio"]
    pf = g["participation_frac"]

    if sr < 0.6:
        regime = "anisotropic"
        expect = "1-bit remax beats remex at 2-3 bits; remex recovers at >=4 bits"
    elif sr < 0.85:
        regime = "mildly-anisotropic"
        expect = "codecs close at low bits; remex likely ahead from 3 bits up"
    else:
        regime = "isotropic"
        expect = "remex dominates at every matched byte budget; 4-bit near-lossless"

    return {
        "regime": regime,
        "expectation": expect,
        "sigma_ratio": sr,
        "participation_frac": pf,
        "centering_matters": g["off_center_ratio"] > 0.15,
    }


def format_report(name: str, g: dict, p: dict) -> str:
    return "\n".join(
        [
            f"## Geometry — {name}",
            "",
            f"| metric | value | isotropic reference |",
            f"|---|--:|--:|",
            f"| n x d | {g['n']} x {g['d']} | — |",
            f"| norm mean | {g['norm_mean']:.4f} | — |",
            f"| norm CV (std/mean) | {g['norm_cv']:.4f} | high = varied lengths |",
            f"| off-center ratio | {g['off_center_ratio']:.4f} | 0 = centered at origin |",
            f"| top-1 eigenvalue frac | {g['eig_top1_frac']:.4f} | {1/g['d']:.4f} |",
            f"| top-10 eigenvalue frac | {g['eig_top10_frac']:.4f} | {10/g['d']:.4f} |",
            f"| participation ratio | {g['participation_ratio']:.1f} / {g['d']} | {g['d']} |",
            f"| post-rotation sigma | {g['post_rot_sigma_mean']:.5f} | {g['post_rot_sigma_expected']:.5f} |",
            f"| **post-rotation sigma ratio** | **{g['post_rot_sigma_ratio']:.4f}** | **1.0000** |",
            f"| post-rotation kurtosis | {g['post_rot_kurtosis']:.3f} | 0.000 |",
            f"| KS reject frac | {g['ks_reject_frac']:.2f} | 0.00 |",
            "",
            f"**Regime: {p['regime']}** (sigma ratio {p['sigma_ratio']:.3f})",
            "",
            f"Prediction: {p['expectation']}",
            "",
            f"Centering expected to matter: {p['centering_matters']}",
        ]
    )
