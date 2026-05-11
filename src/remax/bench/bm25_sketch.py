"""BEIR / NFCorpus BM25 sketch benchmark — issue #36.

Two questions decide whether the sparse-to-sign path ships:

1. **Sketch fidelity** — does sign-packed count-sketch BM25 preserve the
   BM25 top-K ranking? (R@K vs full BM25)
2. **Relevance fidelity** — does the sketch preserve relevance judgments
   well enough to be useful? (NDCG@10 vs qrels)

This module contains the testable primitives (metrics, loaders, pipeline
orchestrator). The ``bench/run_bm25_sketch.py`` shim wraps :func:`main`
for the issue's invocation contract.

Corpus
------
NFCorpus (BEIR), 3.6k docs, ~300 queries with qrels. Tiny, fast, public.
Layout under ``bench/.cache/NFCorpus/`` after ``bench/fetch_nfcorpus.sh``:

::

    corpus.jsonl           one JSON record per doc:    {_id, title, text}
    queries.jsonl          one JSON record per query:  {_id, text}
    qrels/test.tsv         header + ``qid\\tdocid\\tscore`` rows

Pipeline
--------
1. Tokenize corpus + queries (lowercase + whitespace; matches
   ``rank_bm25``'s default tokenization).
2. Build ground-truth BM25 top-K per query with ``rank_bm25.BM25Okapi``.
3. Build :class:`remax.SparseSignBitQuantizer` sketches at
   :data:`SKETCH_KS_DEFAULT`.
4. Per query: ``encode_query → hamming_distances → top-K``.
5. Report:

   * **R@K vs BM25 top-K** — sketch fidelity (purely against the BM25
     ranking it is approximating).
   * **NDCG@10 vs qrels** — relevance fidelity (against human judgments).
   * **Bytes/doc** — the code size at each sketch dim.

Ablations
---------
* ``center=True/False`` — answers the open question from #1 from data.
* ``signs=True/False`` — sketch with signed counts (Charikar–Chen–
  Farach-Colton) vs feature-hashing (Weinberger et al). Without signs
  the cosine of the sketch concentrates around the cosine of the
  original *plus a positive bias from cross-term collisions*, which
  predicts feature-hashing should be worse at small k.

The numbers themselves are NOT a test — bench is an experiment, not a
contract. Plumbing IS tested (``tests/test_bench_bm25_sketch.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse

from remax import SparseSignBitQuantizer, hamming_distances
from remax.bench.eval import recall_at_k as r_at_k  # re-exported for tests

__all__ = [
    "SKETCH_KS_DEFAULT",
    "tokenize",
    "load_corpus_jsonl",
    "load_queries_jsonl",
    "load_qrels_tsv",
    "r_at_k",
    "ndcg_at_k",
    "mean_ndcg_at_k",
    "bm25_topk_full",
    "build_bm25_csr",
    "sketch_topk",
    "run_pipeline",
    "format_bm25_sketch_md",
    "main",
]

# Issue #36 ladder. The smallest entry must be a multiple of 8 (sign-bit
# encoder requirement) and at least 8 so the smoke test can drive it.
SKETCH_KS_DEFAULT: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
DEFAULT_K_TOPK = 100
DEFAULT_K_NDCG = 10
DEFAULT_SEED = 42


# --------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """Lowercase + whitespace split — matches ``rank_bm25``'s default.

    Deliberately *not* stemming or stripping punctuation: the goal is to
    feed identical token streams to both the full-BM25 baseline and the
    sketch encoder so the only variable is the encoding step. Heavier
    preprocessing is fine in production; here it would muddy the
    comparison.
    """
    return text.lower().split()


# --------------------------------------------------------------------- #
# Loaders — BEIR NFCorpus layout
# --------------------------------------------------------------------- #


def load_corpus_jsonl(path: Path) -> dict[str, str]:
    """Read a BEIR-format ``corpus.jsonl`` into ``{doc_id: "title text"}``."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = str(rec["_id"])
            title = rec.get("title", "") or ""
            text = rec.get("text", "") or ""
            out[doc_id] = f"{title} {text}".strip()
    return out


def load_queries_jsonl(path: Path) -> dict[str, str]:
    """Read a BEIR-format ``queries.jsonl`` into ``{query_id: text}``."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["_id"])] = rec.get("text", "") or ""
    return out


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    """Read a BEIR-format qrels TSV into ``{qid: {doc_id: score}}``.

    Expected format (BEIR convention):

    ::

        query-id    corpus-id    score
        Q1          D1           2
        Q1          D2           1

    The header row is detected by the presence of ``query-id`` in the
    first column and skipped silently. Zero-score rows are preserved —
    NDCG treats them as zero-gain regardless, and dropping them here
    would silently hide annotator decisions from upstream consumers.
    """
    out: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            if qid == "query-id":  # header row
                continue
            out.setdefault(qid, {})[did] = int(score)
    return out


# --------------------------------------------------------------------- #
# NDCG
# --------------------------------------------------------------------- #


def ndcg_at_k(
    pred_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    """NDCG@k for one query with exponential-gain DCG.

    Formula (BEIR convention):

        DCG  = Σ_{i=1..k}  (2^rel_i − 1) / log2(i + 1)
        IDCG = DCG computed on the ideal ordering of ``relevance``
        NDCG = DCG / IDCG   (defined as 0 when IDCG == 0)

    Parameters
    ----------
    pred_ids : sequence of str
        Predicted doc IDs, ordered by descending score. Only the first
        ``k`` are consulted.
    relevance : mapping of str → int
        Per-doc graded relevance. Docs absent from this mapping are
        treated as relevance 0.
    k : int
        Cutoff. Must be positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pred_topk = list(pred_ids)[:k]
    dcg = 0.0
    for i, did in enumerate(pred_topk, start=1):
        rel = int(relevance.get(did, 0))
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(
    preds_per_query: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> float:
    """Average NDCG@k over queries that have at least one relevant doc.

    Queries absent from ``qrels`` (or with no graded-positive entries)
    are skipped so they don't drag the mean toward zero — standard BEIR
    practice. If no eligible query exists the function returns 0.0.
    """
    scores: list[float] = []
    for qid, preds in preds_per_query.items():
        rel = qrels.get(qid)
        if not rel:
            continue
        if not any(v > 0 for v in rel.values()):
            continue
        scores.append(ndcg_at_k(preds, rel, k))
    if not scores:
        return 0.0
    return float(np.mean(scores))


# --------------------------------------------------------------------- #
# BM25 ground truth and CSR construction
# --------------------------------------------------------------------- #


def bm25_topk_full(
    tokenized_docs: list[list[str]],
    tokenized_queries: list[list[str]],
    k: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """Top-k document indices per query under full ``rank_bm25.BM25Okapi``.

    Issue #36 fixes BM25 hyperparameters at the library defaults
    (k1=1.2, b=0.75). The bench is *not* tuning BM25 itself.

    Returns
    -------
    np.ndarray, shape (n_queries, k), dtype int64
        Document indices into ``tokenized_docs``.
    """
    from rank_bm25 import BM25Okapi  # local import — optional dep

    bm = BM25Okapi(tokenized_docs, k1=k1, b=b)
    out = np.zeros((len(tokenized_queries), k), dtype=np.int64)
    for qi, q in enumerate(tokenized_queries):
        scores = bm.get_scores(q)
        # argpartition + stable argsort on the candidate set produces a
        # deterministic ordering even when scores tie at the boundary.
        if k < len(scores):
            cand = np.argpartition(-scores, k)[:k]
        else:
            cand = np.arange(len(scores))
        order = np.argsort(-scores[cand], kind="stable")
        out[qi] = cand[order][:k]
    return out


def build_bm25_csr(
    tokenized_docs: list[list[str]],
    tokenized_queries: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, dict[str, int]]:
    """Build BM25 doc CSR and a query CSR aligned to the same vocabulary.

    Uses :func:`remax.bm25.bm25_csr` for the doc-side weights and
    :func:`remax.bm25.bm25_query` for the query side. The query CSR is
    a sparse count vector over the doc-side vocabulary — OOV terms are
    silently dropped, matching ``rank_bm25.BM25Okapi.get_scores``.
    """
    from remax.bm25 import bm25_csr, bm25_query

    docs_csr, vocab = bm25_csr(tokenized_docs, k1=k1, b=b)
    if len(tokenized_queries) == 0:
        empty = scipy.sparse.csr_matrix((0, docs_csr.shape[1]), dtype=np.float64)
        return docs_csr, empty, vocab
    rows = []
    for q in tokenized_queries:
        rows.append(
            bm25_query(q, vocab, df=np.zeros(1), N=len(tokenized_docs))
        )
    queries_csr = scipy.sparse.vstack(rows).tocsr()
    return docs_csr, queries_csr, vocab


# --------------------------------------------------------------------- #
# Sketch retrieval
# --------------------------------------------------------------------- #


def _make_encoder(
    d: int,
    k: int,
    seed: int,
    center: bool,
    signs: bool,
) -> SparseSignBitQuantizer:
    """Construct a sparse encoder, optionally with signs disabled.

    ``signs=False`` switches the count-sketch into feature-hashing
    (Weinberger et al.) — sums collisions without sign flips. The
    encoder primitive (#33) does not expose this switch directly, but
    the per-dim sign table is a plain array on the instance, so we
    overwrite it after construction.
    """
    enc = SparseSignBitQuantizer(d=d, k=k, seed=seed, center=center)
    if not signs:
        enc.sign_[:] = 1
    return enc


def sketch_topk(
    encoder: SparseSignBitQuantizer,
    docs_csr: scipy.sparse.csr_matrix,
    queries_csr: scipy.sparse.csr_matrix,
    k: int,
) -> np.ndarray:
    """Encode corpus + queries with ``encoder`` and return Hamming top-k.

    Ties are broken by stable ordering on the index — matches the
    determinism contract checked in
    ``tests/test_bench_bm25_sketch.py::test_smoke_pipeline_is_deterministic``.
    """
    codes = encoder.encode(docs_csr)
    q_codes = encoder.encode(queries_csr)
    out = np.zeros((queries_csr.shape[0], k), dtype=np.int64)
    n = codes.shape[0]
    k_use = min(k, n)
    for qi in range(q_codes.shape[0]):
        dists = hamming_distances(codes, q_codes[qi])
        if k_use < n:
            cand = np.argpartition(dists, k_use)[:k_use]
        else:
            cand = np.arange(n)
        order = np.argsort(dists[cand], kind="stable")
        out[qi, :k_use] = cand[order][:k_use]
    return out


# --------------------------------------------------------------------- #
# Pipeline orchestrator
# --------------------------------------------------------------------- #


def run_pipeline(
    *,
    documents: Sequence[str],
    queries: Sequence[str],
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    sketch_ks: Sequence[int] = SKETCH_KS_DEFAULT,
    k_topk: int = DEFAULT_K_TOPK,
    k_ndcg: int = DEFAULT_K_NDCG,
    seed: int = DEFAULT_SEED,
    center_options: Sequence[bool] = (False, True),
    signs_options: Sequence[bool] = (True, False),
) -> dict:
    """End-to-end NFCorpus BM25 sketch benchmark.

    Parameters
    ----------
    documents, queries
        Raw string corpora; tokenization is applied internally so the
        bench harness owns the comparison contract.
    doc_ids, query_ids
        IDs aligned to ``documents`` / ``queries``.
    qrels
        ``{query_id: {doc_id: graded_relevance}}``.
    sketch_ks
        Sketch dimensions to evaluate. Each must be a multiple of 8.
    k_topk
        Top-k cutoff for the BM25 fidelity comparison (issue #36 default 100).
    k_ndcg
        NDCG cutoff (issue #36 default 10).
    seed
        Seed for the SparseSignBitQuantizer hash stream.
    center_options, signs_options
        Ablation toggles. The Cartesian product over both is run for each
        sketch dim, so the default settings produce 4 rows per dim
        (centering × signs).

    Returns
    -------
    dict
        ``{"config": {...}, "rows": [...]}``. Each row records the
        configuration plus ``r_at_k_vs_bm25``, ``ndcg_at_k_vs_qrels``,
        ``bytes_per_doc``, ``n_docs``, ``n_queries``. The config block
        also reports ``ndcg_full_bm25`` — the NDCG@k_ndcg of the full
        BM25 baseline against qrels, which is the ceiling the sketch is
        approximating.
    """
    if len(documents) != len(doc_ids):
        raise ValueError(
            f"documents and doc_ids length mismatch: "
            f"{len(documents)} vs {len(doc_ids)}"
        )
    if len(queries) != len(query_ids):
        raise ValueError(
            f"queries and query_ids length mismatch: "
            f"{len(queries)} vs {len(query_ids)}"
        )
    for k in sketch_ks:
        if k <= 0 or k % 8 != 0:
            raise ValueError(
                f"sketch_ks must be positive multiples of 8 (got {k})"
            )

    tokenized_docs = [tokenize(s) for s in documents]
    tokenized_queries = [tokenize(s) for s in queries]

    # Ground truth: full BM25 top-k over the corpus.
    bm25_truth_idx = bm25_topk_full(tokenized_docs, tokenized_queries, k=k_topk)

    # Doc CSR and query CSR over the same vocabulary, BM25-weighted on
    # the doc side. Query side is sparse counts — ``weights @ q.T`` then
    # recovers ``BM25Okapi.get_scores`` exactly (see remax.bm25).
    docs_csr, queries_csr, _vocab = build_bm25_csr(
        tokenized_docs, tokenized_queries
    )
    d = docs_csr.shape[1]

    # Full-BM25 baseline NDCG@k vs qrels — the ceiling for the sketch.
    bm25_pred_ids: dict[str, list[str]] = {}
    for qi, qid in enumerate(query_ids):
        bm25_pred_ids[qid] = [doc_ids[idx] for idx in bm25_truth_idx[qi]]
    ndcg_full = mean_ndcg_at_k(bm25_pred_ids, qrels, k=k_ndcg)

    rows: list[dict] = []
    for k_sketch in sketch_ks:
        if k_sketch > k_topk and k_sketch > docs_csr.shape[0]:
            # Asking for more neighbors than corpus exists makes no sense;
            # cap k_topk-equivalent at corpus size silently.
            pass
        for center in center_options:
            for signs in signs_options:
                if d == 0:
                    # Empty vocabulary — degenerate corpus, return zeros.
                    pred_idx = np.zeros(
                        (len(queries), k_topk), dtype=np.int64
                    )
                else:
                    enc = _make_encoder(
                        d=d, k=k_sketch, seed=seed,
                        center=center, signs=signs,
                    )
                    if center:
                        enc.fit(docs_csr)
                    pred_idx = sketch_topk(
                        enc, docs_csr, queries_csr, k=k_topk
                    )

                r = float(r_at_k(pred_idx, bm25_truth_idx, k=k_topk))
                sketch_pred_ids: dict[str, list[str]] = {}
                for qi, qid in enumerate(query_ids):
                    sketch_pred_ids[qid] = [
                        doc_ids[idx] for idx in pred_idx[qi]
                    ]
                ndcg = mean_ndcg_at_k(sketch_pred_ids, qrels, k=k_ndcg)

                rows.append({
                    "k_sketch": int(k_sketch),
                    "center": bool(center),
                    "signs": bool(signs),
                    "r_at_k_vs_bm25": r,
                    "ndcg_at_k_vs_qrels": float(ndcg),
                    "bytes_per_doc": int(k_sketch // 8),
                    "n_docs": int(len(documents)),
                    "n_queries": int(len(queries)),
                })

    config = {
        "k_topk": int(k_topk),
        "k_ndcg": int(k_ndcg),
        "seed": int(seed),
        "sketch_ks": list(sketch_ks),
        "n_docs": int(len(documents)),
        "n_queries": int(len(queries)),
        "vocab_size": int(d),
        "ndcg_full_bm25": float(ndcg_full),
    }
    return {"config": config, "rows": rows}


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


_PREDICTIONS_BLOCK = """\
## Predictions (written before running)

Per issue #36, these were committed before the bench was run:

- Sign-packed sketch at k=256 hits **R@100 ≥ 0.85** vs BM25 top-100.
- Quality plateaus near ``k ≈ log(n)/ε² ≈ 1500`` for ``n=3633`` docs
  at ``ε=0.1`` — degrades gracefully below, not a cliff.
- Centering helps less than on dense embeddings (predict: <0.05 R@100 lift).
- Count-sketch beats feature-hashing at small k; gap closes by k=1024.
- NDCG@10 vs qrels for sketch at k=256 ≥ 0.85 × NDCG@10 of full BM25.

### What kills the thesis

- ``R@100 < 0.7 at k=256`` → JL doesn't carry through for sparse-BM25-
  distributed data; investigate why (heavy-tailed weights, insufficient
  k for sparse inputs, query-side sparsity).
- ``NDCG gap > 50% at k=256`` → BM25 ranking is too sensitive to small
  perturbations for the sketch to be a useful approximation.
"""


def _build_narrative(cfg: dict, rows: list[dict]) -> str:
    """Compare results against the issue #36 predictions and report verdict."""
    # Find the k=256, center=False, signs=True row (the canonical config).
    def find(k: int, center: bool, signs: bool) -> Optional[dict]:
        for r in rows:
            if (
                r["k_sketch"] == k
                and r["center"] == center
                and r["signs"] == signs
            ):
                return r
        return None

    headline = find(256, False, True)
    ndcg_full = cfg["ndcg_full_bm25"]

    lines: list[str] = []
    lines.append("## Verdict")
    lines.append("")
    if headline is None:
        lines.append(
            "Headline (k=256, center=off, signs=on) row not in the run — "
            "the table above is the full record."
        )
        return "\n".join(lines) + "\n"

    r100 = headline["r_at_k_vs_bm25"]
    ndcg_sketch = headline["ndcg_at_k_vs_qrels"]
    ndcg_ratio = (ndcg_sketch / ndcg_full) if ndcg_full > 0 else float("nan")

    lines.append(
        f"Canonical config (k=256, center=off, signs=on): "
        f"R@{cfg['k_topk']} = **{r100:.3f}**, "
        f"NDCG@{cfg['k_ndcg']} = **{ndcg_sketch:.3f}** "
        f"(full-BM25 ceiling {ndcg_full:.3f}, "
        f"ratio {ndcg_ratio:.2f}×)."
    )
    lines.append("")
    lines.append("Against the issue's two thesis-kill thresholds:")
    lines.append("")
    if r100 < 0.7:
        lines.append(
            f"- ❌ **R@{cfg['k_topk']} = {r100:.3f} < 0.70** — JL does not "
            "carry through for sparse-BM25-distributed data at this "
            "sketch dim. Thesis killed on fidelity."
        )
    else:
        lines.append(
            f"- ✅ R@{cfg['k_topk']} = {r100:.3f} ≥ 0.70 — sketch "
            "preserves BM25 top-K above the kill threshold."
        )
    if ndcg_full > 0 and ndcg_ratio < 0.5:
        lines.append(
            f"- ❌ **NDCG ratio = {ndcg_ratio:.2f}× < 0.50** — BM25 "
            "ranking is too sensitive to count-sketch perturbations to "
            "be a useful approximation. Thesis killed on relevance."
        )
    else:
        lines.append(
            f"- ✅ NDCG ratio = {ndcg_ratio:.2f}× ≥ 0.50 — sketch "
            "preserves enough relevance signal to be worth shipping."
        )
    lines.append("")
    lines.append(
        "Whether to ship the sparse-to-sign path follows directly from "
        "the boxes above. Stage-2 rerank (per #36's deferred follow-up) "
        "is only worth investigating if at least the fidelity box "
        "passes — the sketch must produce a useful candidate set before "
        "a rerank can recover the ranking inside it."
    )
    return "\n".join(lines) + "\n"


def format_bm25_sketch_md(
    *,
    result: dict,
    version: str,
    dataset_name: str = "NFCorpus",
    notes: Optional[str] = None,
) -> str:
    """Render :func:`run_pipeline` output as ``BM25_SKETCH.md``.

    Structure: predictions block (frozen, copied from issue #36) →
    protocol block → results table → auto-generated verdict comparing
    results to the issue's two thesis-kill thresholds.
    """
    cfg = result["config"]
    rows = result["rows"]

    lines: list[str] = []
    lines.append(f"## remax v{version} — BM25 sketch on {dataset_name}")
    lines.append("")
    lines.append(_PREDICTIONS_BLOCK.rstrip())
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- **Library version**: remax v{version}")
    lines.append(
        f"- **Corpus**: {dataset_name} — {cfg['n_docs']} docs, "
        f"{cfg['n_queries']} queries with qrels, vocab "
        f"{cfg['vocab_size']}."
    )
    lines.append(
        f"- **Sketch fidelity metric**: R@{cfg['k_topk']} of sketch "
        f"top-{cfg['k_topk']} vs full BM25 top-{cfg['k_topk']}."
    )
    lines.append(
        f"- **Relevance fidelity metric**: NDCG@{cfg['k_ndcg']} vs "
        f"qrels (exponential gain). Full-BM25 baseline NDCG@"
        f"{cfg['k_ndcg']} = **{cfg['ndcg_full_bm25']:.3f}**."
    )
    lines.append(
        "- **BM25 hyperparameters**: k1=1.2, b=0.75 (rank_bm25 defaults — "
        "issue #36 does not tune BM25 itself)."
    )
    lines.append(
        "- **Tokenization**: lowercase + whitespace split, matching "
        "``rank_bm25``'s default."
    )
    lines.append(
        f"- **Encoder seed**: {cfg['seed']} (SparseSignBitQuantizer)."
    )
    lines.append(
        "- **Production note**: real-world pipelines feed BM25 weights "
        "from Elasticsearch / Solr / FTS5 directly via "
        "``SparseSignBitQuantizer.encode_from_postings`` (#35) — no "
        "need to materialize the CSR. The bench harness materializes "
        "for compatibility with ``rank_bm25.BM25Okapi`` as the ground "
        "truth oracle."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| k_sketch | center | signs | bytes/doc | R@%d vs BM25 | NDCG@%d vs qrels |"
        % (cfg["k_topk"], cfg["k_ndcg"])
    )
    lines.append(
        "|---------:|:------:|:-----:|----------:|-------------:|----------------:|"
    )
    for r in rows:
        lines.append(
            "| {ks:>8d} | {ce:^6} | {sg:^5} | {bpd:>9d} | "
            "{ra:>12.3f} | {nd:>15.3f} |".format(
                ks=r["k_sketch"],
                ce="yes" if r["center"] else "no",
                sg="yes" if r["signs"] else "no",
                bpd=r["bytes_per_doc"],
                ra=r["r_at_k_vs_bm25"],
                nd=r["ndcg_at_k_vs_qrels"],
            )
        )
    lines.append("")
    lines.append(_build_narrative(cfg, rows).rstrip())
    lines.append("")
    if notes:
        lines.append(notes.rstrip() + "\n")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------- #


def _resolve_cache_dir(arg: Optional[str]) -> Path:
    if arg is not None:
        return Path(arg).expanduser().resolve()
    here = Path(__file__).resolve()
    return here.parents[3] / "bench" / ".cache" / "NFCorpus"


def _resolve_out_path(arg: Optional[str]) -> Path:
    if arg is not None:
        return Path(arg).resolve()
    here = Path(__file__).resolve()
    return here.parents[3] / "bench" / "results" / "BM25_SKETCH.md"


def _read_version() -> str:
    try:
        import remax
        return remax.__version__
    except Exception:
        return "0.0.0"


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-bm25-sketch",
        description=(
            "BEIR/NFCorpus BM25 sketch benchmark (issue #36). "
            "Reports R@K vs full BM25 and NDCG@10 vs qrels at a ladder "
            "of sketch dims."
        ),
    )
    p.add_argument(
        "--cache-dir", type=str, default=None,
        help="NFCorpus cache directory (default: bench/.cache/NFCorpus/)",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="output path for BM25_SKETCH.md (default: bench/results/BM25_SKETCH.md)",
    )
    p.add_argument(
        "--k-topk", type=int, default=DEFAULT_K_TOPK,
        help=f"R@K cutoff vs BM25 (default: {DEFAULT_K_TOPK})",
    )
    p.add_argument(
        "--k-ndcg", type=int, default=DEFAULT_K_NDCG,
        help=f"NDCG cutoff vs qrels (default: {DEFAULT_K_NDCG})",
    )
    p.add_argument(
        "--sketch-ks", type=int, nargs="+", default=list(SKETCH_KS_DEFAULT),
        help=(
            f"sketch dim ladder (default: {list(SKETCH_KS_DEFAULT)}). "
            "Each must be a positive multiple of 8."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"SparseSignBitQuantizer seed (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--qrels-split", type=str, default="test",
        help="qrels split filename stem under qrels/ (default: test)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    cache_dir = _resolve_cache_dir(args.cache_dir)
    corpus_path = cache_dir / "corpus.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    qrels_path = cache_dir / "qrels" / f"{args.qrels_split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            sys.stderr.write(
                f"error: missing {path}\n"
                f"to fix: bash bench/fetch_nfcorpus.sh\n"
            )
            return 1

    sys.stderr.write(f"[load]  corpus from {corpus_path}\n")
    corpus = load_corpus_jsonl(corpus_path)
    sys.stderr.write(f"[load]  queries from {queries_path}\n")
    queries = load_queries_jsonl(queries_path)
    sys.stderr.write(f"[load]  qrels from {qrels_path}\n")
    qrels = load_qrels_tsv(qrels_path)

    # Keep only queries that have at least one positive-graded qrel.
    eligible_qids = [
        qid for qid in queries
        if qid in qrels and any(v > 0 for v in qrels[qid].values())
    ]
    if not eligible_qids:
        sys.stderr.write("error: no eligible queries (qrels empty?)\n")
        return 1
    sys.stderr.write(
        f"[run]   {len(corpus)} docs, {len(eligible_qids)} qrels-eligible queries\n"
    )

    doc_ids = list(corpus.keys())
    docs = [corpus[did] for did in doc_ids]
    qids = eligible_qids
    query_texts = [queries[qid] for qid in qids]

    result = run_pipeline(
        documents=docs,
        queries=query_texts,
        doc_ids=doc_ids,
        query_ids=qids,
        qrels=qrels,
        sketch_ks=tuple(args.sketch_ks),
        k_topk=args.k_topk,
        k_ndcg=args.k_ndcg,
        seed=args.seed,
    )

    md = format_bm25_sketch_md(
        result=result, version=_read_version()
    )
    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"\nwrote {out_path}\n")
    sys.stderr.write(
        f"full-BM25 NDCG@{args.k_ndcg} ceiling: "
        f"{result['config']['ndcg_full_bm25']:.3f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
