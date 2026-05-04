"""Tests for ``remax.bench.rerank`` — float32-IP and cross-encoder rerankers.

The cross-encoder path is exercised with a stub session/tokenizer so the
tests don't download a model and don't require ``onnxruntime`` /
``tokenizers`` to be installed in the test environment. The real model is
covered indirectly by an integration test in ``test_bench_run_rerank.py``
that auto-skips when the optional deps are missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from remax.bench import rerank
from remax.bench.rerank import (
    CrossEncoderReranker,
    DEFAULT_CROSS_ENCODER,
    float32_ip_rerank,
)


# --------------------------------------------------------------------- #
# float32_ip_rerank
# --------------------------------------------------------------------- #


def _hand_corpus():
    """Tiny deterministic corpus where the IP order is obvious."""
    # 4 unit-ish vectors in 8-d. Query is mostly on dim 0.
    corpus = np.array(
        [
            [1.0, 0, 0, 0, 0, 0, 0, 0],   # idx 0  — perfect alignment
            [0.9, 0.1, 0, 0, 0, 0, 0, 0], # idx 1
            [0.1, 0.9, 0, 0, 0, 0, 0, 0], # idx 2
            [-1.0, 0, 0, 0, 0, 0, 0, 0],  # idx 3  — antipodal
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    return corpus, query


def test_float32_ip_rerank_orders_candidates_by_inner_product():
    corpus, query = _hand_corpus()
    candidates = np.array([3, 0, 2, 1], dtype=np.intp)
    out = float32_ip_rerank(
        query=query, corpus=corpus, candidate_idx=candidates, k=4
    )
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3], dtype=np.intp))


def test_float32_ip_rerank_truncates_to_k():
    corpus, query = _hand_corpus()
    out = float32_ip_rerank(
        query=query,
        corpus=corpus,
        candidate_idx=np.array([3, 0, 2, 1], dtype=np.intp),
        k=2,
    )
    np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.intp))


def test_float32_ip_rerank_k_capped_at_candidate_count():
    """If k > |candidates| we just return all of them in IP order, no error."""
    corpus, query = _hand_corpus()
    out = float32_ip_rerank(
        query=query,
        corpus=corpus,
        candidate_idx=np.array([2, 0], dtype=np.intp),
        k=10,
    )
    # IP-ordered subset: idx 0 then idx 2.
    np.testing.assert_array_equal(out, np.array([0, 2], dtype=np.intp))


def test_float32_ip_rerank_validates_k_positive():
    corpus, query = _hand_corpus()
    with pytest.raises(ValueError, match="k must be positive"):
        float32_ip_rerank(
            query=query,
            corpus=corpus,
            candidate_idx=np.array([0], dtype=np.intp),
            k=0,
        )


def test_float32_ip_rerank_rejects_2d_query():
    corpus, _ = _hand_corpus()
    with pytest.raises(ValueError, match="shape mismatch"):
        float32_ip_rerank(
            query=np.zeros((2, 8), dtype=np.float32),
            corpus=corpus,
            candidate_idx=np.array([0], dtype=np.intp),
            k=1,
        )


def test_float32_ip_rerank_rejects_2d_candidates():
    corpus, query = _hand_corpus()
    with pytest.raises(ValueError, match="candidate_idx must be 1-D"):
        float32_ip_rerank(
            query=query,
            corpus=corpus,
            candidate_idx=np.array([[0, 1]], dtype=np.intp),
            k=1,
        )


# --------------------------------------------------------------------- #
# CrossEncoderReranker — stubbed onnxruntime path
# --------------------------------------------------------------------- #


class _StubEncoding:
    def __init__(self, ids, mask, ttype):
        self.ids = ids
        self.attention_mask = mask
        self.type_ids = ttype


class _StubTokenizer:
    """Mimic the tokenizers.Tokenizer surface that rerank.py uses."""

    def __init__(self):
        self.calls = []

    def enable_padding(self, **kwargs):
        pass

    def enable_truncation(self, **kwargs):
        pass

    def encode_batch(self, pairs):
        self.calls.append(list(pairs))
        encs = []
        for q, d in pairs:
            n = max(1, len(q) + len(d))  # arbitrary, just needs equal-length rows
            encs.append(_StubEncoding([1] * n, [1] * n, [0] * n))
        # Pad to the same length so np.array stacks cleanly
        max_n = max(len(e.ids) for e in encs)
        padded = []
        for e in encs:
            pad = max_n - len(e.ids)
            padded.append(
                _StubEncoding(
                    e.ids + [0] * pad,
                    e.attention_mask + [0] * pad,
                    e.type_ids + [0] * pad,
                )
            )
        return padded


class _StubInput:
    def __init__(self, name):
        self.name = name


class _StubSession:
    """Pretend ONNX session whose logit equals -|len(query)-len(doc)|.

    That makes the doc whose length matches the query the top hit, which is
    a deterministic rerank target the tests can verify.
    """

    def __init__(self):
        self._inputs = [
            _StubInput("input_ids"),
            _StubInput("attention_mask"),
            _StubInput("token_type_ids"),
        ]
        self.run_calls = 0

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        self.run_calls += 1
        # The stub doesn't see the original strings — but encode_batch made
        # row length proportional to len(q)+len(d). Since q is identical
        # within a batch the differences are driven by len(d) alone, which
        # is enough for tests to assert ordering.
        ids = feed["input_ids"]
        # Score = -length (shorter rows score higher). Cast to float32 like
        # the real session.
        lengths = (ids != 0).sum(axis=1).astype(np.float32)
        return [(-lengths).reshape(-1, 1)]


def _install_stubs(monkeypatch):
    """Patch CrossEncoderReranker.prepare to install stubs."""
    def fake_prepare(self):
        if self._sess is not None:
            return self
        self._tok = _StubTokenizer()
        self._sess = _StubSession()
        self._input_names = [i.name for i in self._sess.get_inputs()]
        return self

    monkeypatch.setattr(CrossEncoderReranker, "prepare", fake_prepare)


def test_cross_encoder_reranker_validates_constructor_args():
    with pytest.raises(ValueError, match="max_length"):
        CrossEncoderReranker(max_length=0)
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(batch_size=0)


def test_cross_encoder_reranker_default_model_id():
    ce = CrossEncoderReranker()
    assert ce.model_id == DEFAULT_CROSS_ENCODER


def test_cross_encoder_score_pairs_empty_returns_empty(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker()
    out = ce.score_pairs([])
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_cross_encoder_score_pairs_runs_in_batches(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker(batch_size=2)
    pairs = [("q", f"d{i}") for i in range(5)]
    scores = ce.score_pairs(pairs)
    assert scores.shape == (5,)
    # batch_size=2 → 3 sessions runs (2 + 2 + 1).
    assert ce._sess.run_calls == 3


def test_cross_encoder_rerank_orders_by_descending_score(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker(batch_size=8)
    # Stub scores higher for shorter docs (-length).
    candidate_idx = np.array([10, 11, 12], dtype=np.intp)
    candidate_texts = ["loooooooooong", "med", "x"]  # idx 12 (shortest) wins
    out = ce.rerank(
        query_text="q",
        candidate_idx=candidate_idx,
        candidate_texts=candidate_texts,
        k=3,
    )
    # Expected order: shortest (12) > med (11) > long (10).
    np.testing.assert_array_equal(out, np.array([12, 11, 10], dtype=np.intp))


def test_cross_encoder_rerank_truncates_to_k(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker()
    candidate_idx = np.array([10, 11, 12], dtype=np.intp)
    candidate_texts = ["loooooooong", "med", "x"]
    out = ce.rerank(
        query_text="q",
        candidate_idx=candidate_idx,
        candidate_texts=candidate_texts,
        k=2,
    )
    np.testing.assert_array_equal(out, np.array([12, 11], dtype=np.intp))


def test_cross_encoder_rerank_validates_k_positive(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker()
    with pytest.raises(ValueError, match="k must be positive"):
        ce.rerank(
            query_text="q",
            candidate_idx=np.array([0], dtype=np.intp),
            candidate_texts=["x"],
            k=0,
        )


def test_cross_encoder_rerank_rejects_misaligned_texts(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker()
    with pytest.raises(ValueError, match="must align"):
        ce.rerank(
            query_text="q",
            candidate_idx=np.array([0, 1], dtype=np.intp),
            candidate_texts=["only one"],
            k=1,
        )


def test_cross_encoder_rerank_handles_empty_candidate_set(monkeypatch):
    _install_stubs(monkeypatch)
    ce = CrossEncoderReranker()
    out = ce.rerank(
        query_text="q",
        candidate_idx=np.array([], dtype=np.intp),
        candidate_texts=[],
        k=5,
    )
    assert out.shape == (0,)
