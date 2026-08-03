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


# ── Shared-kernel refactor: characterize routes through remax.packing ─────────
#
# characterize.py used to carry a private _POPCOUNT_LUT, its own _sign_pack and
# its own XOR/popcount loop, importing nothing from packing — so characterize()
# ran the NumPy fallback on the largest grids in the codebase while the native
# POPCNT kernel sat unused. These pin that it now delegates, and that
# delegating did not move any number.

import importlib

import numpy as np
import pytest

# NB: `import remax.characterize as ch` binds the *function*, not the module —
# remax/__init__.py does `from .characterize import characterize`, which
# rebinds the attribute on the package object. import_module gets the module.
ch = importlib.import_module("remax.characterize")
pk = importlib.import_module("remax.packing")


def test_sign_pack_matches_encode_signs_when_divisible_by_8():
    """The only sanctioned difference is padding; with no pad they must agree.

    Anchored on packing.encode_signs, which characterize.py did not produce —
    not on a stored golden from characterize's own previous output.
    """
    rng = np.random.default_rng(11)
    for d in (8, 64, 256, 768):
        X = rng.standard_normal((37, d)).astype(np.float32)
        assert np.array_equal(ch._sign_pack(X), pk.encode_signs(X))


def test_sign_pack_pads_ragged_dims_that_encode_signs_rejects():
    """The pad is characterize's own concession and must survive the refactor."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal((5, 13)).astype(np.float32)
    packed = ch._sign_pack(X)
    assert packed.shape == (5, 2)  # 13 dims -> 16 bits -> 2 bytes
    with pytest.raises(ValueError):
        pk.encode_signs(X)  # unpadded, this is an error by design
    # A zero pad packs to sign bit 0 on both sides, so it adds a constant 0 to
    # every distance and cannot reorder anything.
    padded = np.pad(X, ((0, 0), (0, 3)))
    assert np.array_equal(packed, pk.encode_signs(padded))


def test_hamming_topN_uses_the_shared_distance_kernel():
    """Fail if the private popcount loop is ever reintroduced.

    Equal *results* would not catch that — a copied LUT gives identical
    answers, which is exactly how the duplication survived. So this asserts
    the call actually happens.
    """
    rng = np.random.default_rng(13)
    codes = ch._sign_pack(rng.standard_normal((64, 32)).astype(np.float32))
    queries = ch._sign_pack(rng.standard_normal((3, 32)).astype(np.float32))

    calls = []
    real = pk.hamming_distances

    def spy(c, q):
        calls.append((c.shape, q.shape))
        return real(c, q)

    ch.hamming_distances = spy
    try:
        ch._hamming_topN(queries, codes, 10)
    finally:
        ch.hamming_distances = real

    assert len(calls) == 3, (
        "characterize._hamming_topN did not route through "
        "packing.hamming_distances — the private popcount path is back"
    )


def test_characterize_module_has_no_private_popcount_table():
    """The duplicated table itself, named. Cheap and unambiguous."""
    assert not hasattr(ch, "_POPCOUNT_LUT"), (
        "characterize.py has re-grown a private popcount LUT; import "
        "POPCOUNT_LUT (or hamming_distances) from remax.packing instead"
    )


def test_topN_identical_with_and_without_the_native_kernel():
    """Native and fallback must agree — the refactor put native on this path.

    Ties are the norm at integer Hamming distances, so this is a real
    constraint on the selection, not just on the distances.
    """
    rng = np.random.default_rng(14)
    codes = ch._sign_pack(rng.standard_normal((900, 128)).astype(np.float32))
    queries = ch._sign_pack(rng.standard_normal((12, 128)).astype(np.float32))

    import remax._native as native

    with_native = ch._hamming_topN(queries, codes, 50)
    prev = native.AVAILABLE
    native.AVAILABLE = False
    try:
        without = ch._hamming_topN(queries, codes, 50)
    finally:
        native.AVAILABLE = prev

    assert np.array_equal(with_native, without)


def test_characterize_report_unchanged_across_the_kernel_switch():
    """End-to-end: the whole report is identical either way.

    This is the property the refactor claimed. Computed twice in-process
    rather than compared to a checked-in golden, because a golden produced by
    the code under test is a changelog, not an oracle.
    """
    from remax.characterize import characterize

    import remax._native as native

    rng = np.random.default_rng(15)
    corpus = rng.standard_normal((400, 64)).astype(np.float32)
    corpus += rng.standard_normal(64).astype(np.float32) * 0.7
    queries = rng.standard_normal((40, 64)).astype(np.float32)
    strategies = [
        "sign-raw", "sign-centered", "pca", "haar-trunc",
        "gaussian", "countsketch", "f32-raw", "f32-centered",
    ]
    kw = dict(strategies=strategies, k_values=[32, 64], seed=99)

    native_report = characterize(corpus, queries, **kw)
    prev = native.AVAILABLE
    native.AVAILABLE = False
    try:
        fallback_report = characterize(corpus, queries, **kw)
    finally:
        native.AVAILABLE = prev

    assert native_report.table == fallback_report.table
    assert native_report.best == fallback_report.best
    assert native_report.notes == fallback_report.notes


def test_hamming_topN_clamps_N_to_the_corpus_size():
    """N > n must clamp to n, not index the code-width axis.

    Found by mutation: `min(N, c_codes.shape[0])` -> `shape[1]` survived the
    whole suite, because every other test asks for fewer neighbours than the
    corpus has rows and the two axes are both "big enough" there.
    """
    rng = np.random.default_rng(16)
    codes = ch._sign_pack(rng.standard_normal((5, 64)).astype(np.float32))
    queries = ch._sign_pack(rng.standard_normal((2, 64)).astype(np.float32))
    out = ch._hamming_topN(queries, codes, 100)
    assert out.shape == (2, 5)  # clamped to n=5, not to the 8-byte code width
    for row in out:
        assert sorted(row.tolist()) == list(range(5))


def test_characterize_handles_a_corpus_smaller_than_the_100_neighbour_request():
    """Regression: characterize() crashed on any corpus with n < 100.

    np.argpartition(scores, N) requires kth < len(scores), so once N clamped
    to n the call raised `ValueError: kth(=n) out of bounds (n)`. Reproduced
    against origin/main before the fix; invisible until now because every
    existing test used a corpus larger than the request.
    """
    from remax.characterize import characterize as characterize_fn

    rng = np.random.default_rng(17)
    for n in (12, 60, 99, 100, 101):
        corpus = rng.standard_normal((n, 64)).astype(np.float32)
        queries = rng.standard_normal((5, 64)).astype(np.float32)
        report = characterize_fn(
            corpus, queries, strategies=["sign-raw", "f32-raw"], k_values=[64]
        )
        assert report.table, f"n={n} produced no rows"
        for row in report.table:
            assert 0.0 <= row["R@10"] <= 1.0, f"n={n}: {row}"


def test_float32_topN_clamps_like_hamming_topN():
    """Same latent crash lived in the float32 ground-truth path."""
    rng = np.random.default_rng(18)
    corpus = rng.standard_normal((7, 32)).astype(np.float32)
    queries = rng.standard_normal((3, 32)).astype(np.float32)
    out = ch._float32_topN(queries, corpus, 100)
    assert out.shape == (3, 7)
    for row in out:
        assert sorted(row.tolist()) == list(range(7))
