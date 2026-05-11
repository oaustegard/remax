"""Tests for ``remax.bench.bm25_sketch`` — the NFCorpus BM25 sketch bench.

Required by issue #36:

1. **Metric correctness**: ``r_at_k`` and ``ndcg_at_k`` with hand-computed
   fixed cases (4–5 each).
2. **Loader test**: parse a 3-line synthetic NFCorpus-format qrels TSV +
   JSONL corpus/queries; verify known structure.
3. **End-to-end smoke**: 8 docs, 2 queries, k=8 — full pipeline runs
   without exception, output JSON has the expected schema.
4. **Determinism**: smoke twice → identical metric outputs.

These are *plumbing* tests. The bench numbers themselves are an
experiment, not a contract — they live in BM25_SKETCH.md, not here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from remax.bench.bm25_sketch import (
    SKETCH_KS_DEFAULT,
    load_corpus_jsonl,
    load_qrels_tsv,
    load_queries_jsonl,
    ndcg_at_k,
    r_at_k,
    run_pipeline,
    tokenize,
)


# --------------------------------------------------------------------- #
# r_at_k — hand-computed fixed cases
# --------------------------------------------------------------------- #


def test_r_at_k_perfect_match():
    """Identical pred and truth → 1.0 at any cutoff."""
    truth = np.array([[0, 1, 2, 3]])
    pred = np.array([[0, 1, 2, 3]])
    assert r_at_k(pred, truth, k=4) == pytest.approx(1.0)
    assert r_at_k(pred, truth, k=1) == pytest.approx(1.0)


def test_r_at_k_partial_overlap():
    """Two of four match → 0.5 at k=4."""
    truth = np.array([[0, 1, 2, 3]])
    pred = np.array([[0, 1, 9, 8]])
    assert r_at_k(pred, truth, k=4) == pytest.approx(0.5)


def test_r_at_k_no_overlap_zero():
    truth = np.array([[10, 11, 12]])
    pred = np.array([[0, 1, 2]])
    assert r_at_k(pred, truth, k=3) == pytest.approx(0.0)


def test_r_at_k_set_semantics_ignores_order():
    """Same set, reversed order → still 1.0."""
    truth = np.array([[0, 1, 2, 3]])
    pred = np.array([[3, 2, 1, 0]])
    assert r_at_k(pred, truth, k=4) == pytest.approx(1.0)


def test_r_at_k_averages_across_queries():
    """Two queries with R = 1.0 and 0.0 → mean 0.5."""
    truth = np.array([[0, 1], [2, 3]])
    pred = np.array([[0, 1], [8, 9]])
    assert r_at_k(pred, truth, k=2) == pytest.approx(0.5)


# --------------------------------------------------------------------- #
# ndcg_at_k — hand-computed fixed cases
# --------------------------------------------------------------------- #


def test_ndcg_at_k_perfect_ranking_is_one():
    """Predicted order matches the ideal order by relevance → NDCG = 1.0."""
    rel = {"a": 2, "b": 1, "c": 0}
    pred = ["a", "b", "c"]
    assert ndcg_at_k(pred, rel, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_docs_in_topk_is_zero():
    """Predicted top-k contains no relevant doc → NDCG = 0.0."""
    rel = {"a": 2, "b": 1}
    pred = ["x", "y", "z"]
    assert ndcg_at_k(pred, rel, k=3) == pytest.approx(0.0)


def test_ndcg_at_k_empty_qrels_is_zero():
    """No relevant docs at all → NDCG = 0 (IDCG would be 0; defined as 0)."""
    rel: dict[str, int] = {}
    pred = ["a", "b"]
    assert ndcg_at_k(pred, rel, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_reversed_ranking_hand_computed():
    """One relevant doc (rel=1) at the kth position vs at the top.

    Exponential gain: gain = 2**rel - 1.
    Position discount: log2(i + 1) for 1-indexed position.

    rel = {"a": 1}, pred = ["x", "y", "a"], k=3:
        DCG  = (2^1 - 1)/log2(4) = 1/2 = 0.5
        IDCG = (2^1 - 1)/log2(2) = 1/1 = 1.0
        NDCG = 0.5
    """
    rel = {"a": 1}
    pred = ["x", "y", "a"]
    assert ndcg_at_k(pred, rel, k=3) == pytest.approx(0.5)


def test_ndcg_at_k_graded_two_relevant_hand_computed():
    """Two graded relevances, sub-ideal ordering — hand-computed.

    rel = {"a": 2, "b": 1}; pred = ["b", "a"]; k=2.
        DCG  = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3)
             = 1/1 + 3/log2(3)
             = 1 + 3/1.5849625... ≈ 1 + 1.8927892607 = 2.8927892607
        IDCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3)
             = 3 + 1/log2(3) ≈ 3 + 0.6309297535 = 3.6309297535
        NDCG = DCG / IDCG ≈ 0.7967...
    """
    rel = {"a": 2, "b": 1}
    pred = ["b", "a"]
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(pred, rel, k=2) == pytest.approx(expected)


def test_ndcg_at_k_cutoff_truncates_predictions():
    """A relevant doc beyond k must not contribute to DCG."""
    rel = {"a": 1}
    pred = ["x", "y", "a"]
    # k=2 cuts off before "a" — DCG = 0, NDCG = 0
    assert ndcg_at_k(pred, rel, k=2) == pytest.approx(0.0)


# --------------------------------------------------------------------- #
# Loaders — parse synthetic NFCorpus-format files
# --------------------------------------------------------------------- #


def test_load_qrels_tsv_parses_3_line_synthetic(tmp_path: Path):
    """Standard BEIR qrels: header row + ``qid\\tdocid\\tscore`` data rows."""
    path = tmp_path / "qrels.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "PLAIN-1\tMED-1\t2\n"
        "PLAIN-1\tMED-2\t1\n"
        "PLAIN-2\tMED-3\t1\n",
        encoding="utf-8",
    )
    qrels = load_qrels_tsv(path)
    assert qrels == {
        "PLAIN-1": {"MED-1": 2, "MED-2": 1},
        "PLAIN-2": {"MED-3": 1},
    }


def test_load_qrels_tsv_zero_score_kept_or_dropped_explicitly(tmp_path: Path):
    """Score=0 rows are kept verbatim — the metric layer decides what to do."""
    path = tmp_path / "qrels.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "Q1\tD1\t0\n"
        "Q1\tD2\t1\n",
        encoding="utf-8",
    )
    qrels = load_qrels_tsv(path)
    # Zero-relevance entries are preserved (NDCG treats them as 0-gain anyway).
    assert qrels["Q1"]["D1"] == 0
    assert qrels["Q1"]["D2"] == 1


def test_load_corpus_jsonl(tmp_path: Path):
    """BEIR corpus.jsonl: one JSON record per line with _id/title/text."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"_id": "MED-1", "title": "Apples", "text": "are red"}) + "\n"
        + json.dumps({"_id": "MED-2", "title": "Bananas", "text": "are yellow"}) + "\n",
        encoding="utf-8",
    )
    corpus = load_corpus_jsonl(path)
    assert set(corpus.keys()) == {"MED-1", "MED-2"}
    assert "apples" in corpus["MED-1"].lower()
    assert "red" in corpus["MED-1"].lower()


def test_load_queries_jsonl(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"_id": "Q1", "text": "what is BM25?"}) + "\n"
        + json.dumps({"_id": "Q2", "text": "sparse retrieval"}) + "\n",
        encoding="utf-8",
    )
    queries = load_queries_jsonl(path)
    assert queries == {"Q1": "what is BM25?", "Q2": "sparse retrieval"}


# --------------------------------------------------------------------- #
# tokenize — match rank_bm25 defaults (lowercase + whitespace)
# --------------------------------------------------------------------- #


def test_tokenize_lowercases_and_splits_on_whitespace():
    assert tokenize("Hello, World!") == ["hello,", "world!"]
    assert tokenize("  multiple   spaces  ") == ["multiple", "spaces"]
    assert tokenize("") == []


# --------------------------------------------------------------------- #
# End-to-end smoke — 8 docs, 2 queries, k=8 (smallest valid sketch dim)
# --------------------------------------------------------------------- #


def _smoke_inputs():
    """Tiny fixed corpus/queries/qrels for the smoke run.

    Eight docs, two queries; vocabulary chosen so the queries genuinely
    favor the same docs under BM25 and under the sketch.
    """
    documents = [
        "apple banana cherry",        # 0
        "apple apple apple",          # 1
        "banana banana cherry",       # 2
        "durian elderberry fig",      # 3
        "grape grape grape grape",    # 4
        "honeydew imbe jackfruit",    # 5
        "kiwi lemon mango",           # 6
        "nectarine orange papaya",    # 7
    ]
    queries = [
        "apple banana",   # Q1: doc 0, 1, 2 most relevant
        "grape",          # Q2: doc 4 most relevant
    ]
    qrels = {
        "Q1": {"D0": 2, "D1": 1, "D2": 1},
        "Q2": {"D4": 2},
    }
    doc_ids = [f"D{i}" for i in range(len(documents))]
    query_ids = ["Q1", "Q2"]
    return documents, queries, qrels, doc_ids, query_ids


def test_smoke_pipeline_runs_and_has_expected_schema():
    documents, queries, qrels, doc_ids, query_ids = _smoke_inputs()
    out = run_pipeline(
        documents=documents,
        queries=queries,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        sketch_ks=(8,),
        k_topk=8,
        k_ndcg=4,
        seed=0,
        center_options=(False,),
        signs_options=(True,),
    )
    assert "rows" in out
    assert "config" in out
    assert isinstance(out["rows"], list)
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    for key in (
        "k_sketch",
        "center",
        "signs",
        "r_at_k_vs_bm25",
        "ndcg_at_k_vs_qrels",
        "bytes_per_doc",
        "n_docs",
        "n_queries",
    ):
        assert key in row, f"missing schema key: {key}"
    # bytes_per_doc must equal k_sketch / 8 = 1 here
    assert row["bytes_per_doc"] == 1
    assert row["k_sketch"] == 8
    # Metric values must be in [0, 1]
    assert 0.0 <= row["r_at_k_vs_bm25"] <= 1.0
    assert 0.0 <= row["ndcg_at_k_vs_qrels"] <= 1.0
    # And the full-BM25 baseline NDCG must be reported too
    assert "ndcg_full_bm25" in out["config"]
    assert 0.0 <= out["config"]["ndcg_full_bm25"] <= 1.0


def test_smoke_pipeline_is_deterministic():
    """Run twice → identical numeric outputs (sketch + full BM25 baseline)."""
    documents, queries, qrels, doc_ids, query_ids = _smoke_inputs()
    kwargs = dict(
        documents=documents,
        queries=queries,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        sketch_ks=(8, 16),
        k_topk=8,
        k_ndcg=4,
        seed=0,
        center_options=(False, True),
        signs_options=(True, False),
    )
    a = run_pipeline(**kwargs)
    b = run_pipeline(**kwargs)
    assert a["config"]["ndcg_full_bm25"] == b["config"]["ndcg_full_bm25"]
    assert len(a["rows"]) == len(b["rows"])
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra == rb


def test_sketch_ks_default_is_issue_36_ladder():
    """Issue #36 fixes the sketch-dim ladder: 64, 128, 256, 512, 1024, 2048."""
    assert SKETCH_KS_DEFAULT == (64, 128, 256, 512, 1024, 2048)
