"""Tests for ``remax.bench.run_baseline`` — orchestrator + markdown formatter.

The CLI driver itself is smoke-tested elsewhere; the testable seams are:

  * :func:`compute_baseline_for_embeddings` — given an embeddings array,
    run the full v0.1.0 quantizer ladder (1-bit + stacked k=2,4,8) and
    return one row of the BASELINE.md table as a dict.
  * :func:`format_baseline_md` — render the rows into the markdown table.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from remax.bench.run_baseline import (
    compute_baseline_for_embeddings,
    format_baseline_md,
    LADDER_KS,
)


# --------------------------------------------------------------------- #
# Ladder constant
# --------------------------------------------------------------------- #


def test_ladder_ks_matches_issue_spec():
    """Issue #4 calls out k=2,4,8 explicitly. Pin the constant."""
    assert LADDER_KS == (2, 4, 8)


# --------------------------------------------------------------------- #
# compute_baseline_for_embeddings
# --------------------------------------------------------------------- #


def test_compute_baseline_returns_all_required_columns():
    """Issue #4 specifies columns: dataset, n, d, 1-bit, k=2, k=4, k=8."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((300, 32)).astype(np.float32)
    row = compute_baseline_for_embeddings(
        name="synthetic", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    assert row["dataset"] == "synthetic"
    # n is the corpus size after splitting off queries
    assert row["n"] == 300 - 20
    assert row["d"] == 32
    assert "1-bit" in row
    for k in (2, 4, 8):
        assert f"k={k}" in row


def test_compute_baseline_recall_values_in_unit_interval():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((400, 64)).astype(np.float32)
    row = compute_baseline_for_embeddings(
        name="synthetic", emb=emb, n_queries=25, k_eval=10, seed=42,
    )
    for col in ("1-bit", "k=2", "k=4", "k=8"):
        assert 0.0 <= row[col] <= 1.0


def test_compute_baseline_stacked_beats_single_bit_on_easy_problem():
    """k=8 should outperform 1-bit at d=64 on a moderate-size problem.
    Looser ordering tests live in test_bench_eval; this one validates the
    end-to-end orchestrator wiring without surprises."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((1000, 64)).astype(np.float32)
    row = compute_baseline_for_embeddings(
        name="synthetic", emb=emb, n_queries=50, k_eval=10, seed=42,
    )
    assert row["k=8"] > row["1-bit"]


def test_compute_baseline_deterministic_under_fixed_seed():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 32)).astype(np.float32)
    a = compute_baseline_for_embeddings(
        name="x", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    b = compute_baseline_for_embeddings(
        name="x", emb=emb, n_queries=20, k_eval=10, seed=42,
    )
    for col in ("1-bit", "k=2", "k=4", "k=8"):
        assert a[col] == pytest.approx(b[col])


def test_compute_baseline_d_must_be_divisible_by_8():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 30)).astype(np.float32)  # d=30, %8 != 0
    with pytest.raises(ValueError):
        compute_baseline_for_embeddings(
            name="x", emb=emb, n_queries=10, k_eval=10, seed=42,
        )


# --------------------------------------------------------------------- #
# format_baseline_md
# --------------------------------------------------------------------- #


def _sample_rows():
    return [
        {
            "dataset": "SPECTER2", "n": 9900, "d": 768,
            "1-bit": 0.6351, "k=2": 0.7100, "k=4": 0.7900, "k=8": 0.8500,
        },
        {
            "dataset": "MiniLM-L6-v2", "n": 9900, "d": 384,
            "1-bit": None, "k=2": None, "k=4": None, "k=8": None,
        },
    ]


def test_format_baseline_md_contains_table_header():
    md = format_baseline_md(
        rows=_sample_rows(),
        version="0.1.0",
        n_queries=100,
        k_eval=10,
        seed=42,
    )
    # Header row from issue #4
    assert re.search(r"\|\s*dataset\s*\|", md)
    assert re.search(r"\|\s*1-bit\s*\|", md)
    assert re.search(r"\|\s*k=2\s*\|", md)
    assert re.search(r"\|\s*k=4\s*\|", md)
    assert re.search(r"\|\s*k=8\s*\|", md)


def test_format_baseline_md_renders_floats_three_decimals():
    md = format_baseline_md(
        rows=_sample_rows(), version="0.1.0",
        n_queries=100, k_eval=10, seed=42,
    )
    # 0.6351 should render as 0.635 — within the ±0.01 blog-post tolerance
    assert "0.635" in md


def test_format_baseline_md_renders_none_as_dash():
    md = format_baseline_md(
        rows=_sample_rows(), version="0.1.0",
        n_queries=100, k_eval=10, seed=42,
    )
    # Missing values must be visible as "—" (em-dash) so the row is clearly
    # incomplete rather than 0.0 or empty.
    minilm_line = [ln for ln in md.splitlines() if "MiniLM" in ln][0]
    # Three dashes for k=2,4,8 plus 1-bit = 4 dashes minimum on this line
    assert minilm_line.count("—") >= 4


def test_format_baseline_md_includes_protocol_block():
    """The blog-post tolerance requires the seed + n_queries + k_eval be
    documented inline. Without that, the number is not reproducible."""
    md = format_baseline_md(
        rows=_sample_rows(), version="0.1.0",
        n_queries=100, k_eval=10, seed=42,
    )
    assert "100" in md          # n_queries
    assert "seed" in md.lower() # protocol block
    assert "0.1.0" in md        # version
    assert "R@10" in md or "k=10" in md or "@10" in md  # eval metric


def test_format_baseline_md_starts_with_h2():
    md = format_baseline_md(
        rows=_sample_rows(), version="0.1.0",
        n_queries=100, k_eval=10, seed=42,
    )
    assert md.lstrip().startswith("## ")
