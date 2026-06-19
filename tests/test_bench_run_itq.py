"""Tests for the issue #46 experiment orchestrator (``remax.bench.run_itq``).

Drives :func:`compute_itq_experiment` on small synthetic corpora (no cache
needed) and checks the result schema, value ranges, equal-bit budgeting, the
prefix-slicing equivalence, input validation, and the Markdown renderer.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax.bench.run_itq import (
    LADDER_KS,
    _itq_prefix,
    compute_itq_experiment,
    format_itq_md,
)
from remax import StackedITQQuantizer


def _anisotropic(n: int, d: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    X = rng.multivariate_normal(np.zeros(d), A @ A.T, size=n)
    return X.astype(np.float32)


# --------------------------------------------------------------------- #
# Schema + ranges
# --------------------------------------------------------------------- #
def test_default_ladder_matches_issue_spec():
    assert LADDER_KS == (1, 2, 4, 8)


def test_experiment_returns_expected_schema():
    eval_emb = _anisotropic(600, 64, seed=1)
    xfer_emb = _anisotropic(600, 64, seed=2)
    res = compute_itq_experiment(
        eval_name="EVAL", eval_emb=eval_emb,
        transfer_name="XFER", transfer_emb=xfer_emb,
        ladder_ks=(1, 2), n_queries=30, k_eval=10, seed=0, itq_iters=5,
    )
    assert res["eval"] == "EVAL" and res["transfer"] == "XFER"
    assert res["d"] == 64
    assert len(res["rows"]) == 2
    for r, k in zip(res["rows"], (1, 2)):
        assert r["k"] == k
        assert r["n_bits"] == k * 64  # equal-bit budget per rung
        for method in ("haar", "itq_in", "itq_xfer"):
            m = r[method]
            assert 0.0 <= m["recall"] <= 1.0
            assert -1.0 <= m["tau"] <= 1.0
            assert 0.0 <= m["ndcg"] <= 1.0
            assert m["n_bits"] == k * 64


def test_itq_in_corpus_beats_chance():
    """Sanity floor: a learned, in-corpus code recalls well above random on
    structured data (not a strict baseline comparison — that's the report)."""
    eval_emb = _anisotropic(800, 64, seed=7)
    res = compute_itq_experiment(
        eval_name="E", eval_emb=eval_emb,
        transfer_name="X", transfer_emb=_anisotropic(800, 64, seed=8),
        ladder_ks=(4,), n_queries=40, k_eval=10, seed=0, itq_iters=10,
    )
    assert res["rows"][0]["itq_in"]["recall"] > 0.2


# --------------------------------------------------------------------- #
# Prefix-slicing equivalence
# --------------------------------------------------------------------- #
def test_itq_prefix_matches_standalone_fit():
    X = _anisotropic(500, 64, seed=3)
    full = StackedITQQuantizer(d=64, k=4, seed=9, n_iters=6).fit(X)
    pref = _itq_prefix(full, 2)
    assert pref.k == 2
    assert pref.n_bits == 2 * 64
    np.testing.assert_array_equal(pref.rotations_, full.rotations_[:2])
    # encodes identically to a standalone k=2 fit (seed nesting)
    standalone = StackedITQQuantizer(d=64, k=2, seed=9, n_iters=6).fit(X)
    np.testing.assert_array_equal(pref.encode(X), standalone.encode(X))


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
def test_transfer_dim_mismatch_raises():
    with pytest.raises(ValueError):
        compute_itq_experiment(
            eval_name="E", eval_emb=_anisotropic(200, 64, seed=1),
            transfer_name="X", transfer_emb=_anisotropic(200, 32, seed=2),
            ladder_ks=(1,), n_queries=20, itq_iters=3,
        )


def test_d_not_divisible_by_8_raises():
    with pytest.raises(ValueError):
        compute_itq_experiment(
            eval_name="E", eval_emb=np.zeros((200, 60), dtype=np.float32),
            transfer_name="X", transfer_emb=np.zeros((200, 60), dtype=np.float32),
            ladder_ks=(1,), n_queries=20, itq_iters=3,
        )


def test_non_ascending_ladder_raises():
    with pytest.raises(ValueError):
        compute_itq_experiment(
            eval_name="E", eval_emb=_anisotropic(200, 64, seed=1),
            transfer_name="X", transfer_emb=_anisotropic(200, 64, seed=2),
            ladder_ks=(4, 2), n_queries=20, itq_iters=3,
        )


# --------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------- #
def test_format_itq_md_has_table_and_protocol():
    eval_emb = _anisotropic(400, 64, seed=1)
    res = compute_itq_experiment(
        eval_name="EVAL", eval_emb=eval_emb,
        transfer_name="XFER", transfer_emb=_anisotropic(400, 64, seed=2),
        ladder_ks=(1, 2), n_queries=20, k_eval=10, seed=0, itq_iters=4,
    )
    md = format_itq_md(
        results=[res], version="0.0.0", n_queries=20, seed=0, itq_iters=4
    )
    assert "Issue #46" in md
    assert "eval = EVAL" in md
    assert "itq_xfer R@10" in md
    assert "win" in md and "xfer Δ" in md
