"""Tests for ``remax.bench.run_topn_sweep`` — PR #22 follow-up.

The sweep is a thin loop around :func:`run_rerank_experiment` that hits
each top_n in a ladder and reports recall + latency at each step. These
tests exercise the orchestrator wiring, the CSV / Markdown formatters,
the PNG smoke path, and the CLI guardrails — using the same fake
cross-encoder pattern as the rerank tests so the suite stays offline.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from remax.bench import datasets, run_topn_sweep
from remax.bench.run_topn_sweep import (
    DEFAULT_TOPN_LADDER,
    _parse_top_ns,
    _plateau_top_n,
    format_topn_sweep_md,
    plot_topn_sweep,
    run_topn_sweep as run_topn_sweep_fn,
    write_topn_sweep_csv,
)


# --------------------------------------------------------------------- #
# Fakes — mirror the test_bench_run_rerank conventions
# --------------------------------------------------------------------- #


class _ConstantReranker:
    """Returns the candidate set unchanged."""

    def __init__(self, model_id: str = "fake-constant"):
        self.model_id = model_id

    def prepare(self):
        return self

    def rerank(self, *, query_text, candidate_idx, candidate_texts, k):
        return candidate_idx[:k]


def _synthetic_corpus(n=200, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    texts = [f"doc-{i}" for i in range(n)]
    return X, texts


# --------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------- #


def test_default_ladder_matches_pr_22_followup_spec():
    """PR #22 follow-up calls out 50, 100, 200, 500, 1000 explicitly."""
    assert DEFAULT_TOPN_LADDER == (50, 100, 200, 500, 1000)


# --------------------------------------------------------------------- #
# run_topn_sweep
# --------------------------------------------------------------------- #


def test_sweep_returns_one_result_per_top_n():
    X, texts = _synthetic_corpus(n=200, d=64)
    ces = [_ConstantReranker()]
    out = run_topn_sweep_fn(
        emb=X, texts=texts, top_ns=[10, 20, 30],
        n_queries=10, k_eval=5, seed=0, cross_encoders=ces,
    )
    assert len(out) == 3
    assert [r["top_n"] for r in out] == [10, 20, 30]


def test_sweep_top_n_must_be_at_least_k_eval():
    X, texts = _synthetic_corpus(n=200, d=64)
    with pytest.raises(ValueError, match="every top_n"):
        run_topn_sweep_fn(
            emb=X, texts=texts, top_ns=[3, 10],
            n_queries=5, k_eval=5, seed=0,
            cross_encoders=[_ConstantReranker()],
        )


def test_sweep_rejects_empty_ladder():
    X, texts = _synthetic_corpus(n=50, d=64)
    with pytest.raises(ValueError, match="not be empty"):
        run_topn_sweep_fn(
            emb=X, texts=texts, top_ns=[],
            n_queries=5, k_eval=5, seed=0,
            cross_encoders=[_ConstantReranker()],
        )


def test_sweep_progress_callback_invoked():
    X, texts = _synthetic_corpus(n=200, d=64)
    seen: list[tuple[int, int, int]] = []

    def progress(top_n, idx, total):
        seen.append((top_n, idx, total))

    run_topn_sweep_fn(
        emb=X, texts=texts, top_ns=[10, 20],
        n_queries=10, k_eval=5, seed=0,
        cross_encoders=[_ConstantReranker()], progress=progress,
    )
    assert seen == [(10, 0, 2), (20, 1, 2)]


def test_sweep_recall_is_monotone_for_oracle_rerank():
    """As top_n grows, the candidate set can only widen, so the float32-IP
    rerank R@K must be monotone non-decreasing — this is the recall
    plateau the sweep is designed to expose."""
    X, texts = _synthetic_corpus(n=400, d=64)
    out = run_topn_sweep_fn(
        emb=X, texts=texts, top_ns=[10, 20, 50, 100, 200],
        n_queries=20, k_eval=5, seed=0,
        cross_encoders=[_ConstantReranker()],
    )
    recalls = [r["stage2a_recall_at_k"] for r in out]
    for a, b in zip(recalls, recalls[1:]):
        assert b >= a - 1e-9, recalls


# --------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------- #


def test_write_csv_has_one_row_per_stage_per_top_n(tmp_path):
    X, texts = _synthetic_corpus(n=200, d=64)
    out = run_topn_sweep_fn(
        emb=X, texts=texts, top_ns=[10, 20],
        n_queries=10, k_eval=5, seed=0,
        cross_encoders=[
            _ConstantReranker(model_id="ce-a"),
            _ConstantReranker(model_id="ce-b"),
        ],
    )
    csv_path = tmp_path / "sweep.csv"
    write_topn_sweep_csv(out, csv_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    # stages per top_n: 1 (stage1) + 1 (stage2a) + 2 (stage2b CEs) = 4
    assert len(rows) == 2 * 4
    by_top_n: dict[str, list[dict]] = {}
    for r in rows:
        by_top_n.setdefault(r["top_n"], []).append(r)
    assert sorted(by_top_n.keys()) == ["10", "20"]
    for stages in by_top_n.values():
        kinds = [r["stage"] for r in stages]
        assert kinds.count("stage1") == 1
        assert kinds.count("stage2a") == 1
        assert kinds.count("stage2b") == 2
        # CE model_ids must be present on stage2b rows
        ce_ids = {r["model_id"] for r in stages if r["stage"] == "stage2b"}
        assert ce_ids == {"ce-a", "ce-b"}


# --------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------- #


def _fake_results():
    """Hand-rolled sweep result with 3 top_n steps and 2 cross-encoders.

    Numbers chosen so the plateau detector picks top_n=200 (stage2a R@K
    is constant from 200 onward) and the latency narrative has a clean
    growth ratio.
    """
    return [
        {
            "n_corpus": 9900, "n_queries": 100, "d": 768,
            "top_n": 100, "k_eval": 10,
            "stage1_recall_at_k": 0.50,
            "stage1_recall_at_topn": 0.80,
            "stage2a_recall_at_k": 0.90,
            "stage2a_latency_s_per_q": 0.0001,
            "stage2b_results": [
                {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 "recall_at_k": 0.30, "latency_s_per_q": 5.0},
                {"model_id": "your-org/scibert-msmarco",
                 "recall_at_k": 0.55, "latency_s_per_q": 6.0},
            ],
        },
        {
            "n_corpus": 9900, "n_queries": 100, "d": 768,
            "top_n": 200, "k_eval": 10,
            "stage1_recall_at_k": 0.60,
            "stage1_recall_at_topn": 0.85,
            "stage2a_recall_at_k": 0.98,
            "stage2a_latency_s_per_q": 0.00015,
            "stage2b_results": [
                {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 "recall_at_k": 0.32, "latency_s_per_q": 10.0},
                {"model_id": "your-org/scibert-msmarco",
                 "recall_at_k": 0.60, "latency_s_per_q": 12.0},
            ],
        },
        {
            "n_corpus": 9900, "n_queries": 100, "d": 768,
            "top_n": 500, "k_eval": 10,
            "stage1_recall_at_k": 0.70,
            "stage1_recall_at_topn": 0.90,
            "stage2a_recall_at_k": 0.98,  # plateau
            "stage2a_latency_s_per_q": 0.00030,
            "stage2b_results": [
                {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 "recall_at_k": 0.31, "latency_s_per_q": 25.0},
                {"model_id": "your-org/scibert-msmarco",
                 "recall_at_k": 0.62, "latency_s_per_q": 30.0},
            ],
        },
    ]


def test_md_includes_recall_and_latency_tables():
    md = format_topn_sweep_md(
        results=_fake_results(), dataset="SPECTER2", seed=42,
    )
    assert "Recall vs top_n" in md
    assert "Per-query latency vs top_n" in md
    # Each top_n shows up
    for t in (100, 200, 500):
        assert str(t) in md
    # Both cross-encoders are named
    assert "ms-marco-MiniLM-L-6-v2" in md
    assert "scibert-msmarco" in md


def test_md_calls_out_plateau_top_n():
    md = format_topn_sweep_md(
        results=_fake_results(), dataset="SPECTER2", seed=42,
    )
    # The plateau detector should pick top_n=200 (within 0.005 of best)
    assert "top_n=200" in md
    assert "plateau" in md.lower()


def test_md_reports_cross_encoder_latency_growth():
    md = format_topn_sweep_md(
        results=_fake_results(), dataset="SPECTER2", seed=42,
    )
    # 100 → 500 is a 5× growth in candidates; latency 5s → 25s = 5× ratio
    assert "5×" in md or "5.0×" in md  # candidate count
    assert "latency growth" in md.lower()


def test_plateau_helper_returns_smallest_within_eps():
    rs = _fake_results()
    # 0.90, 0.98, 0.98 — best is 0.98, plateau is the second entry (top_n=200)
    assert _plateau_top_n(rs, "stage2a_recall_at_k", eps=0.005) == 200
    # Tighter eps would still catch top_n=200 (it equals the best)
    assert _plateau_top_n(rs, "stage2a_recall_at_k", eps=0.0) == 200


def test_plateau_helper_handles_empty():
    assert _plateau_top_n([], "stage2a_recall_at_k") is None


# --------------------------------------------------------------------- #
# Plot — smoke test (file written, non-trivial size)
# --------------------------------------------------------------------- #


def test_plot_writes_a_png(tmp_path):
    out = tmp_path / "sweep.png"
    plot_topn_sweep(_fake_results(), path=out)
    assert out.exists()
    # A real matplotlib PNG is at least a few KB; a stub-rendered one
    # would be much smaller. 2 KB is a generous lower bound.
    assert out.stat().st_size > 2_000


def test_plot_rejects_empty_results(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        plot_topn_sweep([], path=tmp_path / "empty.png")


# --------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------- #


def test_parse_top_ns_default():
    assert _parse_top_ns(None) == DEFAULT_TOPN_LADDER
    assert _parse_top_ns([]) == DEFAULT_TOPN_LADDER


def test_parse_top_ns_repeated_and_csv():
    assert _parse_top_ns(["10", "20"]) == (10, 20)
    assert _parse_top_ns(["10,20,30"]) == (10, 20, 30)
    assert _parse_top_ns(["10", "20,30"]) == (10, 20, 30)


def test_parse_top_ns_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        _parse_top_ns(["0"])
    with pytest.raises(ValueError, match="positive"):
        _parse_top_ns(["10,-5"])


# --------------------------------------------------------------------- #
# CLI end-to-end
# --------------------------------------------------------------------- #


def test_main_rejects_dataset_without_texts(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rc = run_topn_sweep.main(["--dataset", "GloVe-300d"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no registered texts cache" in err


def test_main_runs_end_to_end_with_fake_cross_encoder(
    monkeypatch, tmp_path
):
    """Wire datasets to a tmp cache, force the CLI to use a fake CE, and
    verify it writes the three artifacts (md / csv / png)."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    n, d = 300, 768
    emb = rng.standard_normal((n, d)).astype(np.float32)
    texts = [f"doc-{i}" for i in range(n)]
    emb_path = datasets.dataset_path("SPECTER2")
    txt_path = datasets.texts_path("SPECTER2")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    txt_path.write_text(json.dumps(texts), encoding="utf-8")

    monkeypatch.setattr(
        run_topn_sweep, "CrossEncoderReranker",
        lambda **kw: _ConstantReranker(model_id=kw["model_id"]),
    )

    out_dir = tmp_path / "results"
    rc = run_topn_sweep.main([
        "--dataset", "SPECTER2", "--n", str(n),
        "--queries", "10", "--top-n", "10,20",
        "--cross-encoder-model", "org/model-a",
        "--cross-encoder-model", "org/model-b",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "RERANK_topn_sweep.md").exists()
    assert (out_dir / "rerank_topn_sweep.csv").exists()
    assert (out_dir / "rerank_topn_sweep.png").exists()
    text = (out_dir / "RERANK_topn_sweep.md").read_text()
    assert "model-a" in text
    assert "model-b" in text


def test_main_no_cross_encoder_path_skips_real_model(
    monkeypatch, tmp_path
):
    """``--no-cross-encoder`` short-circuits the CE construction so a
    sweep can run without huggingface_hub / onnxruntime installed."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    n, d = 200, 768
    emb = rng.standard_normal((n, d)).astype(np.float32)
    texts = [f"doc-{i}" for i in range(n)]
    emb_path = datasets.dataset_path("SPECTER2")
    txt_path = datasets.texts_path("SPECTER2")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    txt_path.write_text(json.dumps(texts), encoding="utf-8")

    # Sentinel: blow up loudly if the CLI tries to build a real CE.
    def _explode(**kw):
        raise AssertionError(
            f"CrossEncoderReranker should not be constructed in "
            f"--no-cross-encoder mode (got kw={kw})"
        )

    monkeypatch.setattr(run_topn_sweep, "CrossEncoderReranker", _explode)

    out_dir = tmp_path / "results"
    rc = run_topn_sweep.main([
        "--dataset", "SPECTER2", "--n", str(n),
        "--queries", "10", "--top-n", "10,20",
        "--no-cross-encoder",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    text = (out_dir / "RERANK_topn_sweep.md").read_text()
    assert "no-cross-encoder" in text
