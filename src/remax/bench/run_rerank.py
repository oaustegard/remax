"""Stage-2 rerank experiment driver (issue #20).

Question: on the candidate set produced by centred 1-bit Hamming stage 1,
does a cross-encoder rerank recover R@10 better than float32 inner-product
rerank on the same candidates?

Pipeline (per query):
    stage 1   centred 1-bit SimHash → top-N indices (N defaults to 100)
    stage 2a  float32 inner-product rerank of those N → R@10
    stage 2b  cross-encoder rerank of those N         → R@10

Stage 1 R@10 (raw Hamming) and stage 1 R@N (the candidate ceiling — no
reranker can recover what stage 1 dropped) are reported alongside.

Latency is wall-clock per query for stage 2 only (download / model init are
excluded; ``CrossEncoderReranker.prepare`` is called before timing starts).

Output: ``bench/results/RERANK.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np

from remax import SignBitQuantizer
from remax.bench import datasets
from remax.bench.eval import exact_knn, recall_at_k
from remax.bench.rerank import (
    DEFAULT_CROSS_ENCODER,
    CrossEncoderReranker,
    float32_ip_rerank,
)

__all__ = [
    "run_rerank_experiment",
    "format_rerank_md",
    "main",
]

# Defaults aligned with run_baseline.py so the two experiments use the same
# query split and quantizer rotation seed.
DEFAULT_N = 10_000
DEFAULT_N_QUERIES = 100
DEFAULT_K_EVAL = 10
DEFAULT_TOP_N = 100
DEFAULT_SEED = 42
QUERY_SPLIT_SEED = 99


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #


def _split_indices(
    n_total: int, n_queries: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corpus_idx, query_idx) into the original embedding array.

    Mirrors :func:`remax.bench.run_baseline._split_queries` but returns
    indices so that we can keep the texts list aligned with the embeddings.
    """
    if n_queries <= 0 or n_queries >= n_total:
        raise ValueError(
            f"n_queries={n_queries} out of range for n={n_total}"
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    return perm[n_queries:], perm[:n_queries]


def run_rerank_experiment(
    *,
    emb: np.ndarray,
    texts: Sequence[str],
    n_queries: int = DEFAULT_N_QUERIES,
    top_n: int = DEFAULT_TOP_N,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    cross_encoder: Optional[CrossEncoderReranker] = None,
) -> dict:
    """Run the full stage-1 + stage-2a + stage-2b experiment.

    Parameters
    ----------
    emb : (n, d) float array
    texts : sequence of n strings, aligned with ``emb`` rows
    n_queries : int
        Held-out query count; remainder is corpus.
    top_n : int
        Stage-1 candidate set size per query (must be ``≥ k_eval``).
    k_eval : int
        Recall cutoff for stage 2 (and the raw stage-1 R@K row).
    seed : int
        Master seed for the SimHash rotation.
    cross_encoder : CrossEncoderReranker | None
        Pre-built reranker. Defaults to a fresh
        ``CrossEncoderReranker()``. Pass an explicit instance to override
        the model id, batch size, or max length.

    Returns
    -------
    dict
        Keys: ``n_corpus``, ``n_queries``, ``d``, ``top_n``, ``k_eval``,
        ``stage1_recall_at_k``, ``stage1_recall_at_topn``,
        ``stage2a_recall_at_k``, ``stage2b_recall_at_k``,
        ``stage2a_latency_s_per_q``, ``stage2b_latency_s_per_q``,
        ``cross_encoder_model``.
    """
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape}")
    n_total, d = int(emb.shape[0]), int(emb.shape[1])
    if d % 8 != 0:
        raise ValueError(
            f"d={d} not divisible by 8; remax codes are bit-packed bytewise"
        )
    if len(texts) != n_total:
        raise ValueError(
            f"texts ({len(texts)}) must align with emb rows ({n_total})."
        )
    if top_n < k_eval:
        raise ValueError(
            f"top_n={top_n} must be ≥ k_eval={k_eval} (rerank widens or "
            "preserves the candidate set; it can't expand it)."
        )

    corpus_idx, query_idx = _split_indices(n_total, n_queries, QUERY_SPLIT_SEED)
    corpus = emb[corpus_idx]
    queries = emb[query_idx]
    corpus_texts = [texts[i] for i in corpus_idx]
    query_texts = [texts[i] for i in query_idx]
    n_corpus = corpus.shape[0]

    # Float32 ground truth — uncentered, exactly as the baseline does.
    truth = exact_knn(corpus, queries, k=k_eval)

    # Stage 1: centered 1-bit SimHash, top-N Hamming.
    mu = corpus.mean(axis=0)
    corpus_enc = corpus - mu
    queries_enc = queries - mu
    quantizer = SignBitQuantizer(d=d, seed=seed)
    codes = quantizer.encode(corpus_enc)
    stage1_pred = quantizer.search(queries_enc, codes, k=top_n)
    if stage1_pred.ndim == 1:  # single-query edge case
        stage1_pred = stage1_pred[None, :]

    stage1_r_at_k = recall_at_k(stage1_pred, truth, k=k_eval)
    # R@top_n requires truth at top_n (the candidate ceiling). Recompute it.
    truth_topn = exact_knn(corpus, queries, k=top_n)
    stage1_r_at_topn = recall_at_k(stage1_pred, truth_topn, k=top_n)

    # Stage 2a: float32-IP rerank.
    stage2a_pred = np.empty((n_queries, k_eval), dtype=np.intp)
    t0 = time.perf_counter()
    for i in range(n_queries):
        stage2a_pred[i] = float32_ip_rerank(
            query=queries[i],
            corpus=corpus,
            candidate_idx=stage1_pred[i],
            k=k_eval,
        )
    stage2a_total = time.perf_counter() - t0
    stage2a_r_at_k = recall_at_k(stage2a_pred, truth, k=k_eval)

    # Stage 2b: cross-encoder rerank.
    if cross_encoder is None:
        cross_encoder = CrossEncoderReranker()
    cross_encoder.prepare()  # exclude one-time model load from latency

    stage2b_pred = np.empty((n_queries, k_eval), dtype=np.intp)
    t0 = time.perf_counter()
    for i in range(n_queries):
        cand = stage1_pred[i]
        cand_texts = [corpus_texts[int(j)] for j in cand]
        stage2b_pred[i] = cross_encoder.rerank(
            query_text=query_texts[i],
            candidate_idx=cand,
            candidate_texts=cand_texts,
            k=k_eval,
        )
    stage2b_total = time.perf_counter() - t0
    stage2b_r_at_k = recall_at_k(stage2b_pred, truth, k=k_eval)

    return {
        "n_corpus": n_corpus,
        "n_queries": int(n_queries),
        "d": d,
        "top_n": int(top_n),
        "k_eval": int(k_eval),
        "stage1_recall_at_k": float(stage1_r_at_k),
        "stage1_recall_at_topn": float(stage1_r_at_topn),
        "stage2a_recall_at_k": float(stage2a_r_at_k),
        "stage2b_recall_at_k": float(stage2b_r_at_k),
        "stage2a_latency_s_per_q": float(stage2a_total / n_queries),
        "stage2b_latency_s_per_q": float(stage2b_total / n_queries),
        "cross_encoder_model": cross_encoder.model_id,
    }


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


def _fmt_pct(v: float) -> str:
    return f"{v:.3f}"


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.1f} ms"


def format_rerank_md(
    *,
    result: Mapping,
    dataset: str,
    seed: int,
) -> str:
    """Render a single experiment result as RERANK.md."""
    lines: List[str] = []
    lines.append(
        f"## remax stage-2 rerank — sign-bit candidates → cross-encoder vs float32-IP"
    )
    lines.append("")
    lines.append(f"- **Dataset**: {dataset}  ")
    lines.append(
        f"- **Corpus / queries**: n_corpus={result['n_corpus']}, "
        f"n_queries={result['n_queries']}, d={result['d']}"
    )
    lines.append(
        f"- **Stage 1**: centered 1-bit SimHash, Hamming top-{result['top_n']}. "
        f"Quantizer seed = {seed}, query split seed = {QUERY_SPLIT_SEED}."
    )
    lines.append(
        f"- **Stage 2a**: float32 inner-product rerank (the baseline)."
    )
    lines.append(
        f"- **Stage 2b**: cross-encoder rerank — `{result['cross_encoder_model']}` "
        f"via ONNX Runtime CPU."
    )
    lines.append(
        f"- **Metric**: R@{result['k_eval']} vs float32 inner-product ground "
        f"truth on the raw (un-centered) corpus."
    )
    lines.append(
        f"- **Latency**: wall-clock per query for stage-2 work only "
        f"(stage-1 Hamming and one-time cross-encoder model load are excluded)."
    )
    lines.append("")
    lines.append(
        "### Recall and latency"
    )
    lines.append("")
    lines.append("| stage                        | R@10  | latency / query |")
    lines.append("|------------------------------|-------|-----------------|")
    lines.append(
        f"| stage 1 (raw 1-bit Hamming)  | "
        f"{_fmt_pct(result['stage1_recall_at_k'])} | (n/a)           |"
    )
    lines.append(
        f"| stage 2a (float32-IP rerank) | "
        f"{_fmt_pct(result['stage2a_recall_at_k'])} | "
        f"{_fmt_ms(result['stage2a_latency_s_per_q']):<15s} |"
    )
    lines.append(
        f"| stage 2b (cross-encoder)     | "
        f"{_fmt_pct(result['stage2b_recall_at_k'])} | "
        f"{_fmt_ms(result['stage2b_latency_s_per_q']):<15s} |"
    )
    lines.append("")
    lines.append(
        f"**Stage-1 candidate ceiling (R@{result['top_n']}):** "
        f"{_fmt_pct(result['stage1_recall_at_topn'])} — the upper bound any "
        f"stage-2 reranker can hit on this candidate set. Stage-2 R@10 cannot "
        f"exceed this number; the gap to 1.000 is what stage 1 lost."
    )
    lines.append("")
    lines.append("### Reading the table")
    lines.append("")
    lines.append(
        "Stage 2a (float32-IP) is the cheap baseline. Stage 2b (cross-encoder) "
        "is the expensive treatment. The interesting comparison is whether "
        "stage 2b lifts R@10 above stage 2a, and whether the latency cost is "
        "justified for the recall gain."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _resolve_results_path(out: Optional[str]) -> Path:
    if out is not None:
        return Path(out).resolve()
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "bench" / "results" / "RERANK.md"


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-rerank",
        description=(
            "Cross-encoder vs float32-IP rerank on sign-bit stage-1 candidates."
        ),
    )
    p.add_argument(
        "--dataset", type=str, default="SPECTER2",
        help="text-bearing dataset to run (default: SPECTER2)",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"max embeddings to load (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--queries", type=int, default=DEFAULT_N_QUERIES,
        help=f"held-out query count (default: {DEFAULT_N_QUERIES})",
    )
    p.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"stage-1 candidate set size (default: {DEFAULT_TOP_N})",
    )
    p.add_argument(
        "--k-eval", type=int, default=DEFAULT_K_EVAL,
        help=f"stage-2 R@K cutoff (default: {DEFAULT_K_EVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"quantizer rotation seed (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--cross-encoder-model", type=str, default=DEFAULT_CROSS_ENCODER,
        help=f"HF Hub model id (default: {DEFAULT_CROSS_ENCODER})",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="cross-encoder batch size (default: 32)",
    )
    p.add_argument(
        "--max-length", type=int, default=512,
        help="cross-encoder max token length (default: 512)",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="output markdown path (default: bench/results/RERANK.md)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    spec = datasets.dataset_spec(args.dataset)  # validates name
    if not spec.has_texts:
        sys.stderr.write(
            f"error: dataset {args.dataset!r} has no registered texts cache; "
            f"the cross-encoder rerank experiment needs source text.\n"
        )
        return 2

    sys.stderr.write(f"[load] {args.dataset}: embeddings + texts\n")
    emb, einfo = datasets.load_dataset(args.dataset, n=args.n)
    texts, _ = datasets.load_texts(args.dataset, n=einfo["n"])

    sys.stderr.write(
        f"[run]  n={einfo['n']} d={einfo['dim']} "
        f"queries={args.queries} top_n={args.top_n} k_eval={args.k_eval}\n"
    )
    sys.stderr.write(
        f"[ce]   {args.cross_encoder_model} (preparing ONNX session)\n"
    )

    ce = CrossEncoderReranker(
        model_id=args.cross_encoder_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    ce.prepare()

    result = run_rerank_experiment(
        emb=emb,
        texts=texts,
        n_queries=args.queries,
        top_n=args.top_n,
        k_eval=args.k_eval,
        seed=args.seed,
        cross_encoder=ce,
    )

    sys.stderr.write(
        f"       stage1 R@{result['k_eval']}={result['stage1_recall_at_k']:.3f}  "
        f"stage1 R@{result['top_n']}={result['stage1_recall_at_topn']:.3f}\n"
    )
    sys.stderr.write(
        f"       stage2a R@{result['k_eval']}={result['stage2a_recall_at_k']:.3f}  "
        f"({result['stage2a_latency_s_per_q'] * 1000:.1f} ms/q)\n"
    )
    sys.stderr.write(
        f"       stage2b R@{result['k_eval']}={result['stage2b_recall_at_k']:.3f}  "
        f"({result['stage2b_latency_s_per_q'] * 1000:.1f} ms/q)\n"
    )

    md = format_rerank_md(
        result=result, dataset=args.dataset, seed=args.seed
    )
    out_path = _resolve_results_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"\nwrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
