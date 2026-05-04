"""Stage-2 rerankers for the sign-bit candidate set (issue #20).

Stage 1 of the experiment is centred 1-bit Hamming over the corpus; it
produces a fixed candidate set per query (top-N indices). Stage 2 reorders
that candidate set, and we measure how well each reranker recovers the
true float32 top-K.

Two rerankers ship here:

* :func:`float32_ip_rerank` — float32 inner-product baseline. Identical to
  what the float32 ground truth uses, restricted to the candidate set.
* :class:`CrossEncoderReranker` — pretrained cross-encoder (default
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``) running through ONNX Runtime
  on CPU. No ``torch`` dependency: ``onnxruntime`` + ``tokenizers`` +
  ``huggingface_hub`` only.

The cross-encoder path lazily downloads its ONNX weights and tokenizer
from the Hugging Face Hub on first use. Set ``HF_HOME`` to control the
cache; downloads are ~30 MB for the L-6 default model and persist between
runs.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "float32_ip_rerank",
    "CrossEncoderReranker",
    "DEFAULT_CROSS_ENCODER",
]

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------------------------- #
# Float32-IP rerank (the baseline stage-2 method)
# --------------------------------------------------------------------- #


def float32_ip_rerank(
    *,
    query: np.ndarray,
    corpus: np.ndarray,
    candidate_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """Reorder ``candidate_idx`` by descending float32 inner product.

    Parameters
    ----------
    query : np.ndarray, shape (d,)
        A single query vector. The float32 reranker doesn't batch across
        queries because each query has its own candidate set.
    corpus : np.ndarray, shape (n, d)
        Full corpus embeddings (float32 or float64, will be cast as needed).
    candidate_idx : np.ndarray, shape (c,)
        Indices into ``corpus`` produced by stage 1.
    k : int
        Number of reranked indices to return. Capped at ``len(candidate_idx)``.

    Returns
    -------
    np.ndarray, shape (min(k, c),), dtype intp
        Top-k of ``candidate_idx`` ordered by descending ``corpus[i] @ query``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    candidate_idx = np.asarray(candidate_idx, dtype=np.intp)
    if candidate_idx.ndim != 1:
        raise ValueError(
            f"candidate_idx must be 1-D, got shape {candidate_idx.shape}"
        )
    query = np.asarray(query)
    corpus = np.asarray(corpus)
    if query.ndim != 1 or corpus.ndim != 2 or query.shape[0] != corpus.shape[1]:
        raise ValueError(
            f"shape mismatch: query={query.shape}, corpus={corpus.shape}"
        )
    sub = corpus[candidate_idx]
    scores = sub @ query
    k_eff = min(k, candidate_idx.shape[0])
    if k_eff == candidate_idx.shape[0]:
        order = np.argsort(-scores, kind="stable")
    else:
        part = np.argpartition(-scores, k_eff)[:k_eff]
        order = part[np.argsort(-scores[part], kind="stable")]
    return candidate_idx[order]


# --------------------------------------------------------------------- #
# Cross-encoder rerank
# --------------------------------------------------------------------- #


class CrossEncoderReranker:
    """ONNX-backed cross-encoder rerank for stage 2.

    Loads ``onnx/model.onnx`` and ``tokenizer.json`` from the Hugging Face
    Hub on first use, then scores ``(query_text, doc_text)`` pairs with a
    single ONNX session. Output is the raw logit; higher = more relevant
    (for ms-marco models).

    Parameters
    ----------
    model_id : str
        Hub repo id. Must publish an ``onnx/model.onnx`` and a fast
        ``tokenizer.json``. Defaults to
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M params).
    max_length : int, default=512
        Token cap per pair.
    batch_size : int, default=32
        Pairs per ONNX forward pass. Tune for memory / latency tradeoff.
    onnx_subpath : str, default='onnx/model.onnx'
        Path inside the Hub repo. Switch to ``onnx/model_O3.onnx`` for the
        graph-optimised variant when available.
    cache_dir : str | None
        Forwarded to ``huggingface_hub.hf_hub_download``. ``None`` uses the
        default Hub cache (``HF_HOME`` or ``~/.cache/huggingface``).

    Notes
    -----
    The class deliberately avoids ``torch`` and ``transformers`` to keep
    the bench harness portable. ``onnxruntime`` is CPU-only by default,
    which matches the v0.1.0 anti-goal list (no GPU).
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CROSS_ENCODER,
        max_length: int = 512,
        batch_size: int = 32,
        onnx_subpath: str = "onnx/model.onnx",
        cache_dir: Optional[str] = None,
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.onnx_subpath = onnx_subpath
        self.cache_dir = cache_dir
        self._sess = None
        self._tok = None
        self._input_names: Optional[List[str]] = None

    # -- lazy init -------------------------------------------------------

    def prepare(self) -> "CrossEncoderReranker":
        """Materialise the ONNX session and tokenizer.

        Called automatically on the first ``score_pairs`` / ``rerank`` call;
        invoke explicitly to front-load the (one-time) model download for
        accurate latency measurement.
        """
        if self._sess is not None:
            return self
        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - import-time error path
            raise ImportError(
                "CrossEncoderReranker needs onnxruntime, tokenizers, and "
                "huggingface_hub. Install with `pip install onnxruntime "
                "tokenizers huggingface_hub`."
            ) from e

        onnx_path = hf_hub_download(
            self.model_id, self.onnx_subpath, cache_dir=self.cache_dir
        )
        tok_path = hf_hub_download(
            self.model_id, "tokenizer.json", cache_dir=self.cache_dir
        )
        self._sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._sess.get_inputs()]
        tok = Tokenizer.from_file(tok_path)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        tok.enable_truncation(max_length=self.max_length)
        self._tok = tok
        return self

    # -- scoring ---------------------------------------------------------

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> np.ndarray:
        """Score a flat list of ``(query, doc)`` pairs.

        Returns a ``(len(pairs),)`` float32 array of raw logits. Higher =
        more relevant for ms-marco-trained cross-encoders.
        """
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        self.prepare()
        out = np.empty((len(pairs),), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encs = self._tok.encode_batch(list(batch))
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array(
                    [e.type_ids for e in encs], dtype=np.int64
                )
            logits = self._sess.run(None, feed)[0]
            out[start : start + len(batch)] = logits.reshape(-1)
        return out

    def rerank(
        self,
        *,
        query_text: str,
        candidate_idx: np.ndarray,
        candidate_texts: Iterable[str],
        k: int,
    ) -> np.ndarray:
        """Reorder ``candidate_idx`` by descending cross-encoder logit.

        Parameters
        ----------
        query_text : str
        candidate_idx : np.ndarray, shape (c,)
            Indices into the corpus, returned in the reranked order.
        candidate_texts : iterable of str
            The text for each candidate, aligned positionally with
            ``candidate_idx``.
        k : int
            Top-k cutoff after rerank.

        Returns
        -------
        np.ndarray, shape (min(k, c),), dtype intp
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        candidate_idx = np.asarray(candidate_idx, dtype=np.intp)
        texts = list(candidate_texts)
        if len(texts) != candidate_idx.shape[0]:
            raise ValueError(
                f"candidate_idx ({candidate_idx.shape[0]}) and "
                f"candidate_texts ({len(texts)}) must align."
            )
        if candidate_idx.size == 0:
            return candidate_idx[:0]
        pairs = [(query_text, t) for t in texts]
        scores = self.score_pairs(pairs)
        k_eff = min(k, candidate_idx.shape[0])
        if k_eff == candidate_idx.shape[0]:
            order = np.argsort(-scores, kind="stable")
        else:
            part = np.argpartition(-scores, k_eff)[:k_eff]
            order = part[np.argsort(-scores[part], kind="stable")]
        return candidate_idx[order]
