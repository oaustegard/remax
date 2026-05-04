"""Tests for ``remax.characterize``.

Covers:
1. Basic smoke test: runs without errors on synthetic Gaussian data.
2. Return-type contract: report.best has expected keys; table is a list of dicts.
3. Strategy filtering: only requested strategies appear in the table.
4. k_values filtering: only requested (and valid) k values appear.
5. ground_truth parameter: pre-supplied vs internally computed give comparable results.
6. L2-normalized note detected on unit-normed inputs.
7. Non-normalized note detected on raw (non-unit) inputs.
8. Centering note: sign-centered should help on non-normalized inputs.
9. Invalid strategy raises ValueError.
10. k_values all exceeding d raises ValueError.
11. __str__ renders without error.
12. All registered strategies run without raising.
"""
from __future__ import annotations

import numpy as np
import pytest

from remax import CharacterizeReport, characterize


def _corpus_queries(n: int = 300, q: int = 30, d: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((q, d)).astype(np.float32)
    return corpus, queries


def _unit_corpus_queries(n: int = 300, q: int = 30, d: int = 128, seed: int = 0):
    corpus, queries = _corpus_queries(n, q, d, seed)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return corpus, queries


# ── 1. Basic smoke test ────────────────────────────────────────────────────────

def test_smoke():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64, 128])
    assert isinstance(report, CharacterizeReport)


# ── 2. Return-type contract ────────────────────────────────────────────────────

def test_best_keys():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64])
    required = {"strategy", "k", "R@10", "R@100", "B_vec"}
    assert required.issubset(report.best.keys())


def test_best_values_in_range():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64, 128])
    assert 0.0 <= report.best["R@10"] <= 1.0
    assert 0.0 <= report.best["R@100"] <= 1.0
    assert report.best["B_vec"] > 0
    assert report.best["k"] in {64, 128}


def test_table_is_list_of_dicts():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64])
    assert isinstance(report.table, list)
    assert all(isinstance(row, dict) for row in report.table)


def test_notes_is_str():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64])
    assert isinstance(report.notes, str)


# ── 3. Strategy filtering ──────────────────────────────────────────────────────

def test_strategy_filtering():
    corpus, queries = _corpus_queries()
    strategies = ["sign-raw", "sign-centered"]
    report = characterize(corpus, queries, strategies=strategies, k_values=[64])
    found = {row["strategy"] for row in report.table}
    assert found == set(strategies)


def test_all_default_strategies_in_table():
    corpus, queries = _corpus_queries(d=128)
    report = characterize(corpus, queries, k_values=[64])
    found = {row["strategy"] for row in report.table}
    assert {"sign-raw", "sign-centered", "pca", "haar-trunc"}.issubset(found)


# ── 4. k_values filtering ──────────────────────────────────────────────────────

def test_k_values_respected():
    corpus, queries = _corpus_queries(d=256)
    report = characterize(
        corpus, queries, strategies=["sign-raw"], k_values=[64, 128]
    )
    ks = {row["k"] for row in report.table}
    assert ks == {64, 128}


def test_k_values_exceeding_d_silently_dropped():
    corpus, queries = _corpus_queries(d=64)
    # k=256 exceeds d=64, should be silently ignored
    report = characterize(
        corpus, queries, strategies=["sign-raw"], k_values=[32, 64, 256]
    )
    ks = {row["k"] for row in report.table}
    assert 256 not in ks
    assert {32, 64}.issubset(ks)


# ── 5. ground_truth parameter ──────────────────────────────────────────────────

def test_ground_truth_precomputed():
    corpus, queries = _corpus_queries(n=200, q=20, d=64)
    # Compute ground truth externally
    sims = queries @ corpus.T
    gt = np.argsort(-sims, axis=1)[:, :10].astype(np.intp)

    r1 = characterize(corpus, queries, ground_truth=gt,  strategies=["sign-raw"], k_values=[64])
    r2 = characterize(corpus, queries,                   strategies=["sign-raw"], k_values=[64])
    # Should give the same result since we passed the same top-10 ground truth
    assert r1.best["R@100"] == r2.best["R@100"]


def test_ground_truth_wrong_shape_raises():
    corpus, queries = _corpus_queries(n=200, q=20, d=64)
    bad_gt = np.zeros((15, 10), dtype=np.intp)  # wrong Q
    with pytest.raises(ValueError, match="ground_truth must be"):
        characterize(corpus, queries, ground_truth=bad_gt, k_values=[64])


# ── 6 & 7. L2-norm notes ──────────────────────────────────────────────────────

def test_normalized_note_detected():
    corpus, queries = _unit_corpus_queries()
    report = characterize(corpus, queries, k_values=[64])
    assert "L2-normalized" in report.notes


def test_not_normalized_note_detected():
    corpus, queries = _corpus_queries()
    # Scale corpus to have large norms
    corpus *= 20.0
    report = characterize(corpus, queries, k_values=[64])
    assert "Not L2-normalized" in report.notes


# ── 8. Centering note ─────────────────────────────────────────────────────────

def test_centering_note_present():
    corpus, queries = _corpus_queries()
    # Shift corpus so centering has an obvious effect
    corpus += 5.0
    queries += 5.0
    report = characterize(
        corpus,
        queries,
        strategies=["sign-raw", "sign-centered"],
        k_values=[128],
    )
    assert any(
        w in report.notes
        for w in ("Centering helps", "Centering hurts", "negligible")
    )


# ── 9. Invalid strategy raises ────────────────────────────────────────────────

def test_invalid_strategy_raises():
    corpus, queries = _corpus_queries()
    with pytest.raises(ValueError, match="Unknown strategies"):
        characterize(corpus, queries, strategies=["not-a-strategy"], k_values=[64])


# ── 10. All k exceed d raises ─────────────────────────────────────────────────

def test_all_k_exceed_d_raises():
    corpus, queries = _corpus_queries(d=32)
    with pytest.raises(ValueError, match="No valid k values"):
        characterize(corpus, queries, k_values=[64, 128, 256])


# ── 11. __str__ renders ───────────────────────────────────────────────────────

def test_str_renders():
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, strategies=["sign-centered"], k_values=[64, 128])
    s = str(report)
    assert "Best:" in s
    assert "strategy" in s
    assert "sign-centered" in s


# ── 12. All registered strategies run ────────────────────────────────────────

@pytest.mark.parametrize(
    "strategy",
    [
        "sign-raw",
        "sign-centered",
        "pca",
        "haar-trunc",
        "gaussian",
        "countsketch",
        "f32-raw",
        "f32-centered",
    ],
)
def test_each_strategy_runs(strategy: str):
    corpus, queries = _corpus_queries(n=200, q=20, d=64)
    report = characterize(corpus, queries, strategies=[strategy], k_values=[64])
    assert len(report.table) == 1
    row = report.table[0]
    assert row["strategy"] == strategy
    assert 0.0 <= row["R@10"] <= 1.0
    assert 0.0 <= row["R@100"] <= 1.0
    assert row["B_vec"] > 0


# ── Recall monotonicity ───────────────────────────────────────────────────────

def test_r100_ge_r10():
    """R@100 ≥ R@10 for every row (more candidates can only help or match)."""
    corpus, queries = _corpus_queries()
    report = characterize(corpus, queries, k_values=[64, 128])
    for row in report.table:
        assert row["R@100"] >= row["R@10"] - 1e-9, (
            f"{row['strategy']} k={row['k']}: R@100={row['R@100']} < R@10={row['R@10']}"
        )
