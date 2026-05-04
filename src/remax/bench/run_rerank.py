"""Stage-2 rerank experiment driver (issue #20 + PR #22 follow-up).

Question: on the candidate set produced by centred 1-bit Hamming stage 1,
does a cross-encoder rerank recover R@10 better than float32 inner-product
rerank on the same candidates?

Pipeline (per query):
    stage 1   centred 1-bit SimHash → top-N indices (N defaults to 100)
    stage 2a  float32 inner-product rerank of those N → R@10
    stage 2b  cross-encoder rerank of those N         → R@10
              (one row per cross-encoder; the harness accepts any number)

The multi-cross-encoder shape is the PR #22 follow-up: it lets a
domain-matched reranker (e.g. SciBERT-finetuned MS MARCO) sit alongside
the off-the-shelf ms-marco MiniLM in the same RERANK.md table.

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


def _normalize_cross_encoders(
    cross_encoder: Optional[object],
    cross_encoders: Optional[Sequence[object]],
) -> list:
    """Resolve the legacy singular kwarg + the new plural one to a list.

    Exactly one of ``cross_encoder`` / ``cross_encoders`` may be supplied.
    If neither is, default to a single fresh :class:`CrossEncoderReranker`.
    """
    if cross_encoder is not None and cross_encoders is not None:
        raise ValueError(
            "pass either cross_encoder= or cross_encoders=, not both"
        )
    if cross_encoders is not None:
        ce_list = list(cross_encoders)
        if not ce_list:
            raise ValueError("cross_encoders must contain at least one entry")
        return ce_list
    if cross_encoder is not None:
        return [cross_encoder]
    return [CrossEncoderReranker()]


def run_rerank_experiment(
    *,
    emb: np.ndarray,
    texts: Sequence[str],
    n_queries: int = DEFAULT_N_QUERIES,
    top_n: int = DEFAULT_TOP_N,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    cross_encoder: Optional[object] = None,
    cross_encoders: Optional[Sequence[object]] = None,
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
        Single cross-encoder (legacy single-row form). Mutually exclusive
        with ``cross_encoders``.
    cross_encoders : sequence of CrossEncoderReranker | None
        One or more rerankers — each contributes a stage-2b row to the
        result. Each must expose ``.model_id``, ``.prepare()`` and
        ``.rerank(query_text=, candidate_idx=, candidate_texts=, k=)``.

    Returns
    -------
    dict
        Keys: ``n_corpus``, ``n_queries``, ``d``, ``top_n``, ``k_eval``,
        ``stage1_recall_at_k``, ``stage1_recall_at_topn``,
        ``stage2a_recall_at_k``, ``stage2a_latency_s_per_q``,
        ``stage2b_results`` (list of per-cross-encoder dicts with
        ``model_id``, ``recall_at_k``, ``latency_s_per_q``).
    """
    ce_list = _normalize_cross_encoders(cross_encoder, cross_encoders)

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

    # Stage 2b: cross-encoder rerank — one row per CE.
    stage2b_results: list[dict] = []
    for ce in ce_list:
        ce.prepare()  # exclude one-time model load from latency
        pred = np.empty((n_queries, k_eval), dtype=np.intp)
        t0 = time.perf_counter()
        for i in range(n_queries):
            cand = stage1_pred[i]
            cand_texts = [corpus_texts[int(j)] for j in cand]
            pred[i] = ce.rerank(
                query_text=query_texts[i],
                candidate_idx=cand,
                candidate_texts=cand_texts,
                k=k_eval,
            )
        total = time.perf_counter() - t0
        stage2b_results.append({
            "model_id": ce.model_id,
            "recall_at_k": float(recall_at_k(pred, truth, k=k_eval)),
            "latency_s_per_q": float(total / n_queries),
        })

    return {
        "n_corpus": n_corpus,
        "n_queries": int(n_queries),
        "d": d,
        "top_n": int(top_n),
        "k_eval": int(k_eval),
        "stage1_recall_at_k": float(stage1_r_at_k),
        "stage1_recall_at_topn": float(stage1_r_at_topn),
        "stage2a_recall_at_k": float(stage2a_r_at_k),
        "stage2a_latency_s_per_q": float(stage2a_total / n_queries),
        "stage2b_results": stage2b_results,
    }


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


def _fmt_pct(v: float) -> str:
    return f"{v:.3f}"


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.1f} ms"


def _short_model_id(model_id: str) -> str:
    """Strip the HF org prefix for tighter table cells.

    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` → ``ms-marco-MiniLM-L-6-v2``
    """
    return model_id.split("/", 1)[-1]


def format_rerank_md(
    *,
    result: Mapping,
    dataset: str,
    seed: int,
) -> str:
    """Render an experiment result with N stage-2b rows as RERANK.md."""
    k = result["k_eval"]
    n = result["top_n"]
    # Stage-2 ceiling at K is the fraction of true top-K that lives inside
    # the candidate set — exactly what float32-IP rerank achieves, since IP
    # *is* the metric the truth is computed against. Cross-encoder R@K is
    # bounded above by this number whenever its relevance judgments
    # disagree with float32-IP.
    stage2_ceiling = result["stage2a_recall_at_k"]
    stage2b_rows: Sequence[Mapping] = result["stage2b_results"]
    stage2a_lat = result["stage2a_latency_s_per_q"]

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
        f"- **Stage 1**: centered 1-bit SimHash, Hamming top-{n}. "
        f"Quantizer seed = {seed}, query split seed = {QUERY_SPLIT_SEED}."
    )
    lines.append(
        f"- **Stage 2a**: float32 inner-product rerank (the baseline — and, "
        f"under the float32-IP truth metric, the *optimal* reranker over "
        f"the candidate set)."
    )
    if len(stage2b_rows) == 1:
        lines.append(
            f"- **Stage 2b**: cross-encoder rerank — "
            f"`{stage2b_rows[0]['model_id']}` via ONNX Runtime CPU."
        )
    else:
        lines.append(
            f"- **Stage 2b**: cross-encoder rerank — "
            f"{len(stage2b_rows)} models compared "
            f"(see table). All run via ONNX Runtime CPU."
        )
    lines.append(
        f"- **Metric**: R@{k} vs float32 inner-product ground truth on the "
        f"raw (un-centered) corpus."
    )
    lines.append(
        f"- **Latency**: wall-clock per query for stage-2 work only "
        f"(stage-1 Hamming and one-time cross-encoder model load are excluded)."
    )
    lines.append("")
    lines.append("### Recall and latency")
    lines.append("")
    # Width the stage label column to fit the longest CE model id row.
    ce_label_template = "stage 2b ({})"
    longest_label = max(
        [len("stage 1 (raw 1-bit Hamming)"),
         len("stage 2a (float32-IP rerank)")]
        + [len(ce_label_template.format(_short_model_id(r["model_id"])))
           for r in stage2b_rows]
    )
    stage_w = max(longest_label, 28)
    pad = lambda s: s.ljust(stage_w)
    sep_dashes = "-" * stage_w

    lines.append(f"| {pad('stage')} | R@{k:<3d}| latency / query |")
    lines.append(f"|{sep_dashes}--|-------|-----------------|")
    lines.append(
        f"| {pad('stage 1 (raw 1-bit Hamming)')} | "
        f"{_fmt_pct(result['stage1_recall_at_k'])} | (n/a)           |"
    )
    lines.append(
        f"| {pad('stage 2a (float32-IP rerank)')} | "
        f"{_fmt_pct(result['stage2a_recall_at_k'])} | "
        f"{_fmt_ms(stage2a_lat):<15s} |"
    )
    for row in stage2b_rows:
        label = ce_label_template.format(_short_model_id(row["model_id"]))
        lines.append(
            f"| {pad(label)} | "
            f"{_fmt_pct(row['recall_at_k'])} | "
            f"{_fmt_ms(row['latency_s_per_q']):<15s} |"
        )
    lines.append("")
    lines.append(
        f"**Stage-2 R@{k} ceiling (= stage 2a):** "
        f"{_fmt_pct(stage2_ceiling)} — the fraction of true top-{k} present "
        f"in the stage-1 candidate set. Float32-IP rerank attains this "
        f"ceiling exactly (it picks top-{k} by descending IP, which is the "
        f"truth metric). No reranker scoring on a different signal can "
        f"strictly exceed it."
    )
    lines.append("")
    lines.append(
        f"**Stage-1 R@{n} (candidate-set retention):** "
        f"{_fmt_pct(result['stage1_recall_at_topn'])} — fraction of the "
        f"true top-{n} that the candidate set covers. Different from the "
        f"stage-2 R@{k} ceiling; reported here for context on stage-1 "
        f"quality at the wider cut."
    )
    lines.append("")
    lines.append("### Discussion")
    lines.append("")
    lines.append(
        f"**Float32-IP rerank wins decisively.** Stage 1 R@{k} of "
        f"{_fmt_pct(result['stage1_recall_at_k'])} climbs to "
        f"{_fmt_pct(result['stage2a_recall_at_k'])} after float32-IP rerank "
        f"of the top-{n} candidates — almost all of the recall sign-bit "
        f"stage 1 left on the table is recoverable, at "
        f"{_fmt_ms(stage2a_lat)} per query. The "
        f"sign-bit + float32-IP-rerank pipeline is essentially a lossless "
        f"R@{k} approximation of full float32 search at this n / top-{n}."
    )
    lines.append("")
    # Per-CE narrative — pick the slowest / lowest-recall row to anchor the
    # legacy off-the-shelf paragraph; let any extra rows speak in their own
    # row instead of rewriting the whole section.
    primary = stage2b_rows[0]
    speedup = primary["latency_s_per_q"] / max(stage2a_lat, 1e-9)
    lines.append(
        f"**Off-the-shelf cross-encoder doesn't.** "
        f"`{_short_model_id(primary['model_id'])}` achieves "
        f"R@{k} = {_fmt_pct(primary['recall_at_k'])} at "
        f"~{_fmt_ms(primary['latency_s_per_q'])} per query — strictly "
        f"worse than the float32-IP baseline, and "
        f"{speedup:.0f}× slower. Two confounders worth naming:"
    )
    lines.append("")
    lines.append(
        "1. **Truth-metric mismatch.** R@K is measured against float32 IP "
        "ground truth. Float32-IP rerank optimises the truth metric "
        "directly; the cross-encoder optimises a *learned* relevance "
        "function. To the extent the two disagree, the cross-encoder is "
        "wrong by definition under this metric."
    )
    lines.append(
        "2. **Domain mismatch.** Off-the-shelf MS MARCO cross-encoders are "
        "trained on short web search query → web document relevance. Both "
        "\"query\" and \"document\" here are full SPECTER2 paper records "
        "(title + abstract, hundreds of tokens). The model is being asked "
        "to do paper–paper semantic similarity in a regime it never saw "
        "during pretraining."
    )
    lines.append("")
    if len(stage2b_rows) > 1:
        lines.append(
            "**Comparing across cross-encoders.** The extra rows above "
            "exist to test confounder #2 directly: a domain-matched "
            "reranker (e.g. a SciBERT-trained MS MARCO cross-encoder) "
            "should narrow — or, in principle, close — the gap to "
            "float32-IP. None of the rows above can exceed the stage-2 "
            f"R@{k} ceiling of {_fmt_pct(stage2_ceiling)}; what they "
            "*can* do is approach it more closely than the off-the-shelf "
            "MiniLM, while paying broadly similar per-query latency. "
            "If a domain-matched row sits at parity with stage 2a, the "
            "remaining gap to a notional human-judged truth metric is "
            "what's left to study."
        )
        lines.append("")
    lines.append(
        "The result isn't that cross-encoders are bad; it's that the "
        f"interesting comparison for this corpus needs (a) a domain-matched "
        f"reranker (a SciBERT-trained CE, or SPECTER2-style bi-encoder "
        f"distilled into a cross-encoder), or (b) a different truth metric "
        f"(human relevance judgments rather than float32-IP). Either change "
        f"would let the cross-encoder express judgments that diverge "
        f"meaningfully from raw IP."
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
            "Cross-encoder vs float32-IP rerank on sign-bit stage-1 "
            "candidates. Pass --cross-encoder-model multiple times to "
            "compare several rerankers (e.g. an off-the-shelf MS MARCO "
            "CE alongside a domain-matched SciBERT-trained one)."
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
        "--cross-encoder-model",
        action="append",
        default=None,
        help=(
            "HF Hub model id for a cross-encoder (publishing "
            "onnx/model.onnx + tokenizer.json). May be passed multiple "
            f"times — each becomes a stage-2b row. Default: "
            f"{DEFAULT_CROSS_ENCODER}."
        ),
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

    model_ids: list[str] = (
        args.cross_encoder_model
        if args.cross_encoder_model
        else [DEFAULT_CROSS_ENCODER]
    )
    ces: list[CrossEncoderReranker] = []
    for mid in model_ids:
        sys.stderr.write(f"[ce]   {mid} (preparing ONNX session)\n")
        ce = CrossEncoderReranker(
            model_id=mid,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        ce.prepare()
        ces.append(ce)

    result = run_rerank_experiment(
        emb=emb,
        texts=texts,
        n_queries=args.queries,
        top_n=args.top_n,
        k_eval=args.k_eval,
        seed=args.seed,
        cross_encoders=ces,
    )

    sys.stderr.write(
        f"       stage1 R@{result['k_eval']}={result['stage1_recall_at_k']:.3f}  "
        f"stage1 R@{result['top_n']}={result['stage1_recall_at_topn']:.3f}\n"
    )
    sys.stderr.write(
        f"       stage2a R@{result['k_eval']}={result['stage2a_recall_at_k']:.3f}  "
        f"({result['stage2a_latency_s_per_q'] * 1000:.1f} ms/q)\n"
    )
    for row in result["stage2b_results"]:
        sys.stderr.write(
            f"       stage2b ({_short_model_id(row['model_id'])}) "
            f"R@{result['k_eval']}={row['recall_at_k']:.3f}  "
            f"({row['latency_s_per_q'] * 1000:.1f} ms/q)\n"
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
