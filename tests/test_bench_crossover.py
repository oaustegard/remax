"""Tests for ``remax.bench.crossover`` — issue #5 publication artifact.

The testable seams here are:

  * :func:`compute_crossover_for_embeddings` — runs the matched-bit-budget
    ladder on one embeddings array and returns a list of CrossoverPoints.
  * :func:`check_one_bit_sanity` — flags large b/d=1 deltas.
  * :func:`write_crossover_csv` — tidy CSV for downstream consumers.
  * :func:`format_crossover_md` — narrative + table.
  * :func:`plot_crossover` — PNG smoke test (file exists, non-trivial size).

Real-embedding behaviour (the SPECTER2 inversion, the asymptotic ceiling)
is exercised by manual `python bench/crossover.py` runs against a cache,
not the unit suite — synthetic Gaussian doesn't reproduce the inversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# remex is a bench-only dev dep; skip the whole module if it isn't around so
# pytest runs cleanly on environments that only installed the core package.
pytest.importorskip("remex")
pytest.importorskip("matplotlib")

from remax.bench.crossover import (
    BITS_PER_DIM_LADDER,
    CrossoverPoint,
    METHOD_REMAX,
    METHOD_REMEX,
    REMEX_MAX_BITS,
    check_one_bit_sanity,
    compute_crossover_for_embeddings,
    format_crossover_md,
    plot_crossover,
    write_crossover_csv,
)


# --------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------- #


def test_bits_per_dim_ladder_matches_issue_spec():
    """Issue #5 calls out 1, 2, 3, 4, 6, 8 explicitly."""
    assert BITS_PER_DIM_LADDER == (1, 2, 3, 4, 6, 8)


def test_remex_max_bits_matches_ladder_top():
    """The single-encode-many-precisions path needs bits=8 to reach b/d=8."""
    assert REMEX_MAX_BITS == max(BITS_PER_DIM_LADDER)


# --------------------------------------------------------------------- #
# compute_crossover_for_embeddings
# --------------------------------------------------------------------- #


def _synthetic(seed=0, n=400, d=64):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_compute_crossover_returns_full_grid():
    """One CrossoverPoint per (method, bits_per_dim)."""
    emb = _synthetic()
    points = compute_crossover_for_embeddings(
        name="syn", emb=emb, n_queries=30, k_eval=10, seed=42,
    )
    assert len(points) == 2 * len(BITS_PER_DIM_LADDER)
    methods = {p.method for p in points}
    assert methods == {METHOD_REMAX, METHOD_REMEX}
    bits = {p.bits_per_dim for p in points}
    assert bits == set(BITS_PER_DIM_LADDER)


def test_compute_crossover_all_recalls_in_unit_interval():
    emb = _synthetic()
    points = compute_crossover_for_embeddings(
        name="syn", emb=emb, n_queries=30, k_eval=10, seed=42,
    )
    for p in points:
        assert 0.0 <= p.recall_at_k <= 1.0


def test_compute_crossover_d_must_be_divisible_by_8():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 30)).astype(np.float32)
    with pytest.raises(ValueError):
        compute_crossover_for_embeddings(
            name="x", emb=emb, n_queries=10, k_eval=10, seed=42,
        )


def test_compute_crossover_deterministic_under_fixed_seed():
    emb = _synthetic()
    a = compute_crossover_for_embeddings(
        name="x", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    b = compute_crossover_for_embeddings(
        name="x", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    by_key_a = {(p.method, p.bits_per_dim): p.recall_at_k for p in a}
    by_key_b = {(p.method, p.bits_per_dim): p.recall_at_k for p in b}
    assert by_key_a == pytest.approx(by_key_b)


def test_compute_crossover_remax_curve_monotone_nondecreasing_to_within_noise():
    """remax stacked is supposed to climb monotonically — that's the ladder's
    whole point. We allow a small dip per step (rotation noise across seeds)
    but the k=8 result must beat the k=1 result."""
    emb = _synthetic(n=1000, d=64)
    points = compute_crossover_for_embeddings(
        name="x", emb=emb, n_queries=50, k_eval=10, seed=42,
    )
    by_b = {
        p.bits_per_dim: p.recall_at_k
        for p in points if p.method == METHOD_REMAX
    }
    assert by_b[8] > by_b[1]


# --------------------------------------------------------------------- #
# sanity check
# --------------------------------------------------------------------- #


def test_sanity_check_passes_when_within_tolerance():
    points = [
        CrossoverPoint("d", 1, METHOD_REMAX, 0.50),
        CrossoverPoint("d", 1, METHOD_REMEX, 0.52),
    ]
    [r] = check_one_bit_sanity(points, tol=0.05)
    assert r.passed
    assert r.delta == pytest.approx(-0.02)


def test_sanity_check_fails_when_outside_tolerance():
    points = [
        CrossoverPoint("d", 1, METHOD_REMAX, 0.50),
        CrossoverPoint("d", 1, METHOD_REMEX, 0.80),
    ]
    [r] = check_one_bit_sanity(points, tol=0.05)
    assert not r.passed


def test_sanity_check_skips_dataset_with_only_one_method():
    """If only one side ran (e.g. remex import error on one dataset), the
    sanity check has nothing to compare and should skip rather than crash."""
    points = [CrossoverPoint("d", 1, METHOD_REMAX, 0.5)]
    assert check_one_bit_sanity(points) == []


# --------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------- #


def test_write_crossover_csv_has_header_and_rows(tmp_path: Path):
    points = [
        CrossoverPoint("SPECTER2", 1, METHOD_REMAX, 0.635),
        CrossoverPoint("SPECTER2", 1, METHOD_REMEX, 0.607),
        CrossoverPoint("SPECTER2", 8, METHOD_REMAX, 0.718),
    ]
    out = tmp_path / "results" / "crossover.csv"
    write_crossover_csv(points, out)
    text = out.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "dataset,bits_per_dim,method,R_at_10"
    assert len(lines) == 4
    assert "SPECTER2,1,remax_stacked,0.635000" in text


# --------------------------------------------------------------------- #
# format_crossover_md
# --------------------------------------------------------------------- #


def _two_dataset_points():
    return [
        CrossoverPoint("SPECTER2", 1, METHOD_REMAX, 0.635),
        CrossoverPoint("SPECTER2", 1, METHOD_REMEX, 0.607),
        CrossoverPoint("SPECTER2", 8, METHOD_REMAX, 0.718),
        CrossoverPoint("SPECTER2", 8, METHOD_REMEX, 0.975),
    ]


def test_format_crossover_md_starts_with_h2():
    md = format_crossover_md(
        points=_two_dataset_points(), sanity=[],
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=True, remex_center=False,
    )
    assert md.lstrip().startswith("## ")


def test_format_crossover_md_includes_protocol_block():
    md = format_crossover_md(
        points=_two_dataset_points(), sanity=[],
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=True, remex_center=False,
    )
    assert "100" in md
    assert "seed" in md.lower()
    assert "0.1.0" in md
    assert "preprocessing" in md.lower() or "Preprocessing" in md
    assert "remax" in md
    assert "remex" in md


def test_format_crossover_md_links_to_csv_and_png():
    md = format_crossover_md(
        points=_two_dataset_points(), sanity=[],
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=True, remex_center=False,
        plot_path="crossover.png", csv_path="crossover.csv",
    )
    assert "crossover.png" in md
    assert "crossover.csv" in md


def test_format_crossover_md_renders_sanity_table():
    from remax.bench.crossover import SanityResult
    sanity = [SanityResult("SPECTER2", remax=0.635, remex=0.607, delta=0.028, tol=0.05)]
    md = format_crossover_md(
        points=_two_dataset_points(), sanity=sanity,
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=True, remex_center=False,
    )
    assert "0.635" in md
    assert "0.607" in md
    assert "yes" in md  # passed marker


def test_format_crossover_md_documents_preprocessing_choices():
    md_natural = format_crossover_md(
        points=_two_dataset_points(), sanity=[],
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=True, remex_center=False,
    )
    md_swapped = format_crossover_md(
        points=_two_dataset_points(), sanity=[],
        version="0.1.0", n_queries=100, k_eval=10, seed=42,
        remax_center=False, remex_center=True,
    )
    # Two different protocols should produce different protocol blocks —
    # silently identical text would defeat the documentation purpose.
    assert md_natural != md_swapped


# --------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------- #


def test_plot_crossover_writes_nontrivial_png(tmp_path: Path):
    emb = _synthetic(n=200, d=64)
    points = compute_crossover_for_embeddings(
        name="syn", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    out = tmp_path / "results" / "crossover.png"
    plot_crossover(points, path=out)
    assert out.exists()
    # Empty PNGs are tiny (~100 B). A real plot is multiple kB.
    assert out.stat().st_size > 5_000


def test_plot_crossover_raises_when_no_data(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_crossover([], path=tmp_path / "x.png")
