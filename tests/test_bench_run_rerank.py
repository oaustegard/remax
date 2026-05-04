"""Tests for ``remax.bench.run_rerank`` — orchestrator and CLI.

The orchestrator is exercised end-to-end on a small synthetic corpus with a
fake cross-encoder so the suite stays offline. A real-model integration
test is gated on ``onnxruntime`` import availability.

Multi-cross-encoder shape (PR #22 follow-up): the orchestrator accepts a
list of rerankers and emits one stage-2b row per model. These tests cover
both the legacy single-CE form and the new multi-CE form.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from remax.bench import datasets, run_rerank
from remax.bench.rerank import CrossEncoderReranker
from remax.bench.run_rerank import (
    QUERY_SPLIT_SEED,
    format_rerank_md,
    run_rerank_experiment,
)


# --------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------- #


class _OracleReranker:
    """A fake reranker that returns the float32 ground-truth top-k.

    Used to verify that the orchestrator wires the reranker into the
    candidate pipeline correctly: feeding it a perfect reranker should
    drive stage-2b R@K to the candidate-set ceiling.
    """

    model_id = "fake-oracle"

    def __init__(self, *, corpus, queries, query_idx_map):
        self._corpus = corpus
        self._queries = queries
        self._query_idx_map = query_idx_map  # text → query row index

    def prepare(self):
        return self

    def rerank(self, *, query_text, candidate_idx, candidate_texts, k):
        # Recover the query row from the supplied text. The orchestrator
        # passes the literal text from `query_texts`, so the inversion is
        # exact.
        qrow = self._query_idx_map[query_text]
        scores = self._corpus[candidate_idx] @ self._queries[qrow]
        order = np.argsort(-scores, kind="stable")[:k]
        return candidate_idx[order]


class _ConstantReranker:
    """Fake reranker that returns the candidate set unchanged."""

    def __init__(self, model_id: str = "fake-constant"):
        self.model_id = model_id

    def prepare(self):
        return self

    def rerank(self, *, query_text, candidate_idx, candidate_texts, k):
        return candidate_idx[:k]


class _ReverseReranker:
    """Fake reranker that returns the candidate set reversed."""

    model_id = "fake-reverse"

    def prepare(self):
        return self

    def rerank(self, *, query_text, candidate_idx, candidate_texts, k):
        return candidate_idx[::-1][:k]


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _synthetic_corpus(n=200, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    texts = [f"doc-{i}" for i in range(n)]
    return X, texts


# --------------------------------------------------------------------- #
# run_rerank_experiment — wiring & math
# --------------------------------------------------------------------- #


def test_orchestrator_returns_expected_keys():
    X, texts = _synthetic_corpus()
    ce = _ConstantReranker()
    res = run_rerank_experiment(
        emb=X, texts=texts, n_queries=10, top_n=20, k_eval=5, seed=0,
        cross_encoder=ce,
    )
    expected = {
        "n_corpus", "n_queries", "d", "top_n", "k_eval",
        "stage1_recall_at_k", "stage1_recall_at_topn",
        "stage2a_recall_at_k", "stage2a_latency_s_per_q",
        "stage2b_results",
    }
    assert expected.issubset(res.keys())
    assert res["n_corpus"] == X.shape[0] - 10
    assert res["n_queries"] == 10
    assert res["d"] == X.shape[1]
    assert res["top_n"] == 20
    assert res["k_eval"] == 5
    assert isinstance(res["stage2b_results"], list)
    assert len(res["stage2b_results"]) == 1
    row = res["stage2b_results"][0]
    assert row["model_id"] == "fake-constant"
    assert {"recall_at_k", "latency_s_per_q"}.issubset(row.keys())


def test_orchestrator_accepts_multiple_cross_encoders():
    """The PR #22 follow-up shape: pass cross_encoders=[a, b, c] and get
    one stage-2b row per model in the result."""
    X, texts = _synthetic_corpus(n=200, d=64)
    ces = [
        _ConstantReranker(model_id="ce-a"),
        _ConstantReranker(model_id="ce-b"),
        _ReverseReranker(),
    ]
    res = run_rerank_experiment(
        emb=X, texts=texts, n_queries=10, top_n=20, k_eval=5, seed=0,
        cross_encoders=ces,
    )
    rows = res["stage2b_results"]
    assert [r["model_id"] for r in rows] == ["ce-a", "ce-b", "fake-reverse"]
    # Both constant rerankers should give the same recall (they produce
    # identical orderings); the reverse reranker may differ.
    assert rows[0]["recall_at_k"] == pytest.approx(rows[1]["recall_at_k"])


def test_orchestrator_rejects_both_singular_and_plural():
    X, texts = _synthetic_corpus(n=50, d=64)
    with pytest.raises(ValueError, match="not both"):
        run_rerank_experiment(
            emb=X, texts=texts, n_queries=5, top_n=10, k_eval=5, seed=0,
            cross_encoder=_ConstantReranker(),
            cross_encoders=[_ConstantReranker()],
        )


def test_orchestrator_rejects_empty_cross_encoders():
    X, texts = _synthetic_corpus(n=50, d=64)
    with pytest.raises(ValueError, match="at least one"):
        run_rerank_experiment(
            emb=X, texts=texts, n_queries=5, top_n=10, k_eval=5, seed=0,
            cross_encoders=[],
        )


def test_oracle_reranker_hits_the_candidate_ceiling():
    """A perfect (oracle) reranker should drive stage 2 R@K to the
    candidate-set ceiling — no higher, no lower. This is the cleanest
    possible end-to-end wiring check."""
    X, texts = _synthetic_corpus(n=300, d=64)
    n_queries = 20
    rng = np.random.default_rng(QUERY_SPLIT_SEED)
    perm = rng.permutation(X.shape[0])
    corpus = X[perm[n_queries:]]
    queries = X[perm[:n_queries]]
    query_texts = [texts[i] for i in perm[:n_queries]]
    qmap = {t: i for i, t in enumerate(query_texts)}
    ce = _OracleReranker(corpus=corpus, queries=queries, query_idx_map=qmap)

    res = run_rerank_experiment(
        emb=X, texts=texts, n_queries=n_queries,
        top_n=50, k_eval=10, seed=42, cross_encoder=ce,
    )
    oracle_row = res["stage2b_results"][0]

    # Float32-IP rerank is mathematically equivalent to the oracle for the
    # ranking metric used here (both order candidates by descending IP). So
    # 2a and 2b should match exactly.
    assert res["stage2a_recall_at_k"] == pytest.approx(
        oracle_row["recall_at_k"]
    )
    # And stage 2 cannot fall below stage 1 R@K when the reranker is at
    # least as good as the stage-1 ranking — the oracle/float32 rerank
    # always wins or ties.
    assert res["stage2a_recall_at_k"] >= res["stage1_recall_at_k"] - 1e-9


def test_constant_reranker_preserves_stage1_recall_at_k():
    """If the reranker returns candidate_idx unchanged, stage-2b R@K equals
    stage-1 R@K (the same indices in the same order)."""
    X, texts = _synthetic_corpus(n=200, d=64)
    res = run_rerank_experiment(
        emb=X, texts=texts, n_queries=10, top_n=30, k_eval=10, seed=0,
        cross_encoder=_ConstantReranker(),
    )
    assert res["stage2b_results"][0]["recall_at_k"] == pytest.approx(
        res["stage1_recall_at_k"]
    )


def test_top_n_must_be_at_least_k_eval():
    X, texts = _synthetic_corpus()
    with pytest.raises(ValueError, match="top_n"):
        run_rerank_experiment(
            emb=X, texts=texts, n_queries=5, top_n=4, k_eval=10, seed=0,
            cross_encoder=_ConstantReranker(),
        )


def test_dim_must_be_divisible_by_8():
    X = np.random.default_rng(0).standard_normal((50, 7)).astype(np.float32)
    texts = [f"d{i}" for i in range(50)]
    with pytest.raises(ValueError, match="not divisible by 8"):
        run_rerank_experiment(
            emb=X, texts=texts, n_queries=5, top_n=10, k_eval=5, seed=0,
            cross_encoder=_ConstantReranker(),
        )


def test_texts_must_align_with_embeddings():
    X, texts = _synthetic_corpus(n=50)
    with pytest.raises(ValueError, match="align"):
        run_rerank_experiment(
            emb=X, texts=texts[:10], n_queries=5, top_n=10, k_eval=5, seed=0,
            cross_encoder=_ConstantReranker(),
        )


def test_latency_fields_are_finite_and_positive():
    X, texts = _synthetic_corpus()
    res = run_rerank_experiment(
        emb=X, texts=texts, n_queries=5, top_n=10, k_eval=5, seed=0,
        cross_encoder=_ConstantReranker(),
    )
    assert np.isfinite(res["stage2a_latency_s_per_q"])
    assert res["stage2a_latency_s_per_q"] >= 0.0
    for row in res["stage2b_results"]:
        assert np.isfinite(row["latency_s_per_q"])
        assert row["latency_s_per_q"] >= 0.0


# --------------------------------------------------------------------- #
# format_rerank_md
# --------------------------------------------------------------------- #


def _fake_result(stage2b_results=None):
    if stage2b_results is None:
        stage2b_results = [{
            "model_id": "ce/test",
            "recall_at_k": 0.75,
            "latency_s_per_q": 0.05,
        }]
    return {
        "n_corpus": 9900,
        "n_queries": 100,
        "d": 768,
        "top_n": 100,
        "k_eval": 10,
        "stage1_recall_at_k": 0.5,
        "stage1_recall_at_topn": 0.85,
        "stage2a_recall_at_k": 0.7,
        "stage2a_latency_s_per_q": 0.0001,
        "stage2b_results": stage2b_results,
    }


def test_format_rerank_md_includes_all_recall_numbers():
    md = format_rerank_md(result=_fake_result(), dataset="SPECTER2", seed=42)
    assert "0.500" in md   # stage1 R@K
    assert "0.850" in md   # stage1 R@top_n
    assert "0.700" in md   # stage2a
    assert "0.750" in md   # stage2b
    assert "SPECTER2" in md
    assert "ce/test" in md or "test" in md  # short id is OK


def test_format_rerank_md_includes_latency_in_ms():
    md = format_rerank_md(result=_fake_result(), dataset="SPECTER2", seed=42)
    assert "0.1 ms" in md   # 1e-4 s
    assert "50.0 ms" in md  # 5e-2 s


def test_format_rerank_md_labels_ceiling_as_stage2a():
    """The R@K stage-2 ceiling equals stage-2a's R@K (float32-IP rerank
    is optimal under the float32-IP truth metric). The markdown must say
    so — the previous wording mislabelled R@top_n as the ceiling."""
    md = format_rerank_md(result=_fake_result(), dataset="SPECTER2", seed=42)
    assert "Stage-2 R@10 ceiling" in md
    assert "stage 2a" in md
    # And R@top_n is reported separately, not as the ceiling
    assert "candidate-set retention" in md


def test_format_rerank_md_includes_discussion():
    md = format_rerank_md(result=_fake_result(), dataset="SPECTER2", seed=42)
    assert "Discussion" in md
    # Speedup ratio between the two stage-2 latencies must appear
    assert "500" in md  # 5e-2 / 1e-4 = 500x


def test_format_rerank_md_renders_one_row_per_cross_encoder():
    """Multi-CE results: one stage-2b row per model in the table."""
    rows = [
        {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
         "recall_at_k": 0.305, "latency_s_per_q": 5.0},
        {"model_id": "cross-encoder/scibert-msmarco",
         "recall_at_k": 0.65, "latency_s_per_q": 6.5},
    ]
    md = format_rerank_md(
        result=_fake_result(stage2b_results=rows),
        dataset="SPECTER2", seed=42,
    )
    # Both models must appear in the table
    assert "ms-marco-MiniLM-L-6-v2" in md
    assert "scibert-msmarco" in md
    # And each recall number is present
    assert "0.305" in md
    assert "0.650" in md
    # Multi-CE narrative paragraph mentions the comparison
    assert "domain-matched" in md.lower()


# --------------------------------------------------------------------- #
# CLI guardrails
# --------------------------------------------------------------------- #


def test_main_rejects_dataset_without_texts(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rc = run_rerank.main(["--dataset", "GloVe-300d"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no registered texts cache" in err


def test_main_rejects_unknown_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        run_rerank.main(["--dataset", "not-a-real-dataset"])


def test_main_runs_end_to_end_with_fake_cross_encoder(
    monkeypatch, tmp_path, capsys
):
    """Wire datasets to a tmp cache, force the CLI to use a fake CE, and
    verify it writes a RERANK.md."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    n, d = 200, 64
    emb = rng.standard_normal((n, d)).astype(np.float32)
    texts = [f"doc-{i}" for i in range(n)]
    emb_path = datasets.dataset_path("SPECTER2")
    txt_path = datasets.texts_path("SPECTER2")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb.reshape(n, d).repeat(768 // d, axis=1)[:, :768])
    txt_path.write_text(json.dumps(texts), encoding="utf-8")

    # Replace CrossEncoderReranker with the constant fake so the CLI's
    # construction path doesn't try to download a real model.
    monkeypatch.setattr(
        run_rerank, "CrossEncoderReranker",
        lambda **kw: _ConstantReranker(model_id=kw.get("model_id", "fake")),
    )

    out_path = tmp_path / "RERANK.md"
    rc = run_rerank.main(
        ["--dataset", "SPECTER2", "--n", str(n),
         "--queries", "10", "--top-n", "20",
         "--out", str(out_path)]
    )
    assert rc == 0
    assert out_path.exists()
    text = out_path.read_text()
    assert "SPECTER2" in text
    assert "stage 2a" in text
    assert "stage 2b" in text


def test_main_accepts_multiple_cross_encoder_models(
    monkeypatch, tmp_path
):
    """``--cross-encoder-model`` can be passed repeatedly; one stage-2b row
    per model lands in the markdown."""
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

    monkeypatch.setattr(
        run_rerank, "CrossEncoderReranker",
        lambda **kw: _ConstantReranker(model_id=kw["model_id"]),
    )

    out_path = tmp_path / "RERANK.md"
    rc = run_rerank.main([
        "--dataset", "SPECTER2", "--n", str(n),
        "--queries", "10", "--top-n", "20",
        "--cross-encoder-model", "org/model-a",
        "--cross-encoder-model", "org/model-b",
        "--out", str(out_path),
    ])
    assert rc == 0
    text = out_path.read_text()
    # Short-form ids appear in the table
    assert "model-a" in text
    assert "model-b" in text
