"""Top-N sweep experiment driver (PR #22 follow-up).

For a fixed query set, vary the stage-1 candidate-set size ``top_n`` over
a ladder (default 50, 100, 200, 500, 1000) and measure stage-2a / stage-2b
R@K and per-query latency at each step.

Two questions the sweep answers:

1. **Where does float32-IP rerank plateau?** — R@K vs top_n flattens once
   the candidate set covers the true top-K with high probability. Every
   bit of work past that point is wasted recall headroom for negligible
   gain.
2. **Where does cross-encoder latency become prohibitive?** — CE latency
   is ~linear in ``top_n`` (per-pair forward pass). The sweep lets you
   pick the operating point on the recall/latency curve directly rather
   than guessing.

Output triple: ``RERANK_topn_sweep.md`` (narrative + tables),
``rerank_topn_sweep.csv`` (tidy long format), and
``rerank_topn_sweep.png`` (two-panel plot).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np

from remax.bench import datasets
from remax.bench.rerank import (
    DEFAULT_CROSS_ENCODER,
    CrossEncoderReranker,
)
from remax.bench.run_rerank import (
    DEFAULT_K_EVAL,
    DEFAULT_N,
    DEFAULT_N_QUERIES,
    DEFAULT_SEED,
    QUERY_SPLIT_SEED,
    _short_model_id,
    run_rerank_experiment,
)

__all__ = [
    "DEFAULT_TOPN_LADDER",
    "run_topn_sweep",
    "format_topn_sweep_md",
    "plot_topn_sweep",
    "write_topn_sweep_csv",
    "main",
]

# Ladder pinned by the PR #22 follow-up: 50 / 100 / 200 / 500 / 1000. Wide
# enough to show the float32-IP recall plateau and to make CE latency
# growth obvious; not so wide that a single SPECTER2 sweep takes hours.
DEFAULT_TOPN_LADDER: tuple[int, ...] = (50, 100, 200, 500, 1000)


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #


def run_topn_sweep(
    *,
    emb: np.ndarray,
    texts: Sequence[str],
    top_ns: Sequence[int] = DEFAULT_TOPN_LADDER,
    n_queries: int = DEFAULT_N_QUERIES,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    cross_encoders: Optional[Sequence[object]] = None,
    progress: Optional[callable] = None,
) -> list[dict]:
    """Run the rerank experiment once per ``top_n`` value.

    Parameters
    ----------
    emb, texts, n_queries, k_eval, seed
        Forwarded to :func:`run_rerank_experiment`.
    top_ns
        Sequence of stage-1 candidate-set sizes to evaluate. All entries
        must be ``≥ k_eval``.
    cross_encoders
        Forwarded as-is to each rerank invocation. Pass an
        already-prepared list so the model load happens once, not per
        ``top_n``.
    progress
        Optional callable ``progress(top_n, idx, total)`` invoked before
        each step. Useful for CLI logging.

    Returns
    -------
    list of dict
        One entry per ``top_n`` — each is the same shape as
        :func:`run_rerank_experiment` returns.
    """
    top_ns = list(top_ns)
    if not top_ns:
        raise ValueError("top_ns must not be empty")
    if any(t < k_eval for t in top_ns):
        raise ValueError(
            f"every top_n must be ≥ k_eval ({k_eval}); got {top_ns}"
        )

    results: list[dict] = []
    for i, t in enumerate(top_ns):
        if progress is not None:
            progress(t, i, len(top_ns))
        res = run_rerank_experiment(
            emb=emb, texts=texts,
            n_queries=n_queries, top_n=t, k_eval=k_eval, seed=seed,
            cross_encoders=cross_encoders,
        )
        results.append(res)
    return results


# --------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------- #


def write_topn_sweep_csv(
    results: Sequence[Mapping], path: Path
) -> None:
    """Write tidy long-format CSV.

    Columns: ``top_n, stage, model_id, recall_at_k, latency_s_per_q``.
    Stage values are ``stage1``, ``stage2a``, ``stage2b``. ``model_id``
    is empty for stage 1 / stage 2a; the latency column is empty for
    stage 1.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "top_n", "stage", "model_id", "recall_at_k", "latency_s_per_q",
        ])
        for r in results:
            t = r["top_n"]
            w.writerow([t, "stage1", "",
                        f"{r['stage1_recall_at_k']:.6f}", ""])
            w.writerow([t, "stage2a", "",
                        f"{r['stage2a_recall_at_k']:.6f}",
                        f"{r['stage2a_latency_s_per_q']:.6f}"])
            for row in r["stage2b_results"]:
                w.writerow([
                    t, "stage2b", row["model_id"],
                    f"{row['recall_at_k']:.6f}",
                    f"{row['latency_s_per_q']:.6f}",
                ])


# --------------------------------------------------------------------- #
# PNG plot
# --------------------------------------------------------------------- #


def plot_topn_sweep(
    results: Sequence[Mapping],
    *,
    path: Path,
    figsize: tuple[float, float] = (12.0, 4.5),
) -> None:
    """Render the two-panel sweep plot.

    Left panel: R@K vs ``top_n`` — one line per stage (stage1, stage2a,
    plus one per stage-2b model). Linear x-axis (the ladder is small).

    Right panel: per-query latency vs ``top_n``, log y-scale to handle
    the ~3-4 order-of-magnitude gap between stage-2a (~0.1 ms) and
    cross-encoders (~seconds).

    The plot is the visual answer to the PR #22 follow-up: where does
    the float32-IP curve plateau, and where does CE latency become
    prohibitive?
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("no results to plot")

    top_ns = [r["top_n"] for r in results]
    stage1_y = [r["stage1_recall_at_k"] for r in results]
    stage2a_y = [r["stage2a_recall_at_k"] for r in results]
    stage2a_lat = [r["stage2a_latency_s_per_q"] for r in results]

    # Discover the cross-encoder ids from the first result (they must be
    # consistent across rows; we don't enforce it here, just iterate).
    ce_ids = [row["model_id"] for row in results[0]["stage2b_results"]]
    ce_recall: dict[str, list[float]] = {m: [] for m in ce_ids}
    ce_latency: dict[str, list[float]] = {m: [] for m in ce_ids}
    for r in results:
        for row in r["stage2b_results"]:
            mid = row["model_id"]
            ce_recall.setdefault(mid, []).append(row["recall_at_k"])
            ce_latency.setdefault(mid, []).append(row["latency_s_per_q"])

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=figsize)

    # --- Left panel: recall ---
    ax_r.plot(top_ns, stage1_y, marker="o", label="stage 1 (1-bit Hamming)")
    ax_r.plot(
        top_ns, stage2a_y, marker="s", linewidth=2.0,
        label="stage 2a (float32-IP rerank)",
    )
    for mid, ys in ce_recall.items():
        ax_r.plot(
            top_ns, ys, marker="^", linestyle="--",
            label=f"stage 2b ({_short_model_id(mid)})",
        )
    ax_r.set_xlabel("top_n (stage-1 candidate set size)")
    ax_r.set_ylabel("R@K vs float32 IP truth")
    ax_r.set_title("Recall vs candidate-set size")
    ax_r.set_xticks(top_ns)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", fontsize=8)

    # --- Right panel: latency (log-y) ---
    ax_l.plot(
        top_ns, [s * 1000 for s in stage2a_lat], marker="s", linewidth=2.0,
        label="stage 2a (float32-IP rerank)",
    )
    for mid, lats in ce_latency.items():
        ax_l.plot(
            top_ns, [s * 1000 for s in lats], marker="^", linestyle="--",
            label=f"stage 2b ({_short_model_id(mid)})",
        )
    ax_l.set_xlabel("top_n (stage-1 candidate set size)")
    ax_l.set_ylabel("latency / query (ms, log scale)")
    ax_l.set_title("Latency vs candidate-set size")
    ax_l.set_xticks(top_ns)
    ax_l.set_yscale("log")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Stage-2 rerank: top-N sweep — recall plateau vs latency growth"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Markdown narrative
# --------------------------------------------------------------------- #


def _fmt_pct(v: float) -> str:
    return f"{v:.3f}"


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.1f} ms"


def _plateau_top_n(
    results: Sequence[Mapping], key: str, eps: float = 0.005
) -> Optional[int]:
    """Return the smallest ``top_n`` whose ``key`` is within ``eps`` of
    the best value seen across the sweep.

    Used to pick out the operating point at which extra candidates buy
    less than ``eps`` recall. Returns ``None`` if the sweep is empty.
    """
    if not results:
        return None
    best = max(r[key] for r in results)
    for r in results:
        if best - r[key] <= eps:
            return r["top_n"]
    return None


def format_topn_sweep_md(
    *,
    results: Sequence[Mapping],
    dataset: str,
    seed: int,
    plot_path: str = "rerank_topn_sweep.png",
    csv_path: str = "rerank_topn_sweep.csv",
) -> str:
    """Render the sweep narrative + recall + latency tables."""
    if not results:
        raise ValueError("results is empty — nothing to render")

    k = results[0]["k_eval"]
    n_queries = results[0]["n_queries"]
    n_corpus = results[0]["n_corpus"]
    d = results[0]["d"]
    ce_ids = [row["model_id"] for row in results[0]["stage2b_results"]]

    plateau = _plateau_top_n(results, "stage2a_recall_at_k", eps=0.005)

    lines: List[str] = []
    lines.append("## remax stage-2 rerank — top-N sweep")
    lines.append("")
    lines.append(
        "How does the stage-1 candidate-set size interact with stage-2 "
        "recall and latency? This sweep answers two questions raised in "
        "PR #22's follow-up list: where the float32-IP rerank R@K plateau "
        "lies, and where cross-encoder latency becomes prohibitive."
    )
    lines.append("")
    lines.append(f"![sweep plot]({plot_path})")
    lines.append("")
    lines.append(f"Tidy CSV: [`{csv_path}`]({csv_path}).")
    lines.append("")

    lines.append(f"- **Dataset**: {dataset}")
    lines.append(
        f"- **Corpus / queries**: n_corpus={n_corpus}, n_queries={n_queries}, d={d}"
    )
    lines.append(
        f"- **Stage 1**: centered 1-bit SimHash, Hamming top-N. Quantizer "
        f"seed = {seed}, query split seed = {QUERY_SPLIT_SEED}."
    )
    lines.append(
        f"- **Stage 2a**: float32 inner-product rerank "
        f"(optimal under the float32-IP truth metric)."
    )
    if ce_ids:
        joined = ", ".join(f"`{_short_model_id(m)}`" for m in ce_ids)
        lines.append(
            f"- **Stage 2b**: cross-encoder rerank — {joined} via "
            f"ONNX Runtime CPU."
        )
    lines.append(
        f"- **Metric**: R@{k} vs float32 inner-product ground truth on the "
        f"raw (un-centered) corpus."
    )
    lines.append(
        f"- **Latency**: wall-clock per query for stage-2 work only "
        f"(stage-1 Hamming and one-time CE model load are excluded)."
    )
    lines.append("")

    # Recall table
    lines.append("### Recall vs top_n")
    lines.append("")
    header = ["top_n", "stage 1", "stage 2a"] + [
        f"stage 2b ({_short_model_id(m)})" for m in ce_ids
    ]
    sep = ["-" * len(h) for h in header]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for r in results:
        row = [
            str(r["top_n"]),
            _fmt_pct(r["stage1_recall_at_k"]),
            _fmt_pct(r["stage2a_recall_at_k"]),
        ]
        ce_row = {
            x["model_id"]: x["recall_at_k"]
            for x in r["stage2b_results"]
        }
        for m in ce_ids:
            v = ce_row.get(m)
            row.append(_fmt_pct(v) if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Latency table
    lines.append("### Per-query latency vs top_n")
    lines.append("")
    header = ["top_n", "stage 2a"] + [
        f"stage 2b ({_short_model_id(m)})" for m in ce_ids
    ]
    sep = ["-" * len(h) for h in header]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for r in results:
        row = [str(r["top_n"]), _fmt_ms(r["stage2a_latency_s_per_q"])]
        ce_row = {
            x["model_id"]: x["latency_s_per_q"]
            for x in r["stage2b_results"]
        }
        for m in ce_ids:
            v = ce_row.get(m)
            row.append(_fmt_ms(v) if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Discussion
    lines.append("### Discussion")
    lines.append("")
    if plateau is not None:
        lines.append(
            f"**Float32-IP plateau.** The stage-2a R@{k} curve flattens at "
            f"top_n={plateau} (within 0.005 of the best point in the sweep). "
            f"Past that, extra candidates only widen the recall ceiling that "
            f"stage-2a is already pinning — wasted work for negligible gain "
            f"under the float32-IP truth metric."
        )
        lines.append("")
    if ce_ids:
        # Approximate per-pair latency, growth ratio across the ladder
        first = results[0]
        last = results[-1]
        for mid in ce_ids:
            f_lat = next(
                x["latency_s_per_q"]
                for x in first["stage2b_results"] if x["model_id"] == mid
            )
            l_lat = next(
                x["latency_s_per_q"]
                for x in last["stage2b_results"] if x["model_id"] == mid
            )
            ratio_t = last["top_n"] / max(first["top_n"], 1)
            ratio_l = l_lat / max(f_lat, 1e-9)
            lines.append(
                f"**`{_short_model_id(mid)}` latency growth.** Going from "
                f"top_n={first['top_n']} to top_n={last['top_n']} "
                f"({ratio_t:.0f}× more candidates) costs {ratio_l:.1f}× "
                f"more time per query "
                f"({_fmt_ms(f_lat)} → {_fmt_ms(l_lat)}). Cross-encoder "
                f"work scales ~linearly in candidate count: every extra "
                f"100 candidates is ~"
                f"{((l_lat - f_lat) / max(last['top_n'] - first['top_n'], 1) * 1000 * 100):.1f} ms "
                f"of extra latency at this batch size."
            )
            lines.append("")
    lines.append(
        f"**Operating-point takeaway.** For SPECTER2-shaped corpora, "
        f"sign-bit + float32-IP rerank at top_n in the low hundreds is a "
        f"near-lossless R@{k} approximation of full float32 search at "
        f"sub-millisecond per-query cost. Any cross-encoder rerank, "
        f"whether off-the-shelf or domain-matched, pays a cost that grows "
        f"linearly with top_n; pick the smallest top_n that hits the "
        f"recall target you need, then stop."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _resolve_results_dir(out: Optional[str]) -> Path:
    if out is not None:
        return Path(out).resolve()
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "bench" / "results"


def _parse_top_ns(values: Optional[Sequence[str]]) -> tuple[int, ...]:
    if not values:
        return DEFAULT_TOPN_LADDER
    out: list[int] = []
    for v in values:
        for piece in v.split(","):
            piece = piece.strip()
            if not piece:
                continue
            t = int(piece)
            if t <= 0:
                raise ValueError(f"top_n must be positive, got {t}")
            out.append(t)
    if not out:
        raise ValueError("--top-n produced no values")
    return tuple(out)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-rerank-topn-sweep",
        description=(
            "Sweep stage-1 candidate-set size and chart how stage-2 "
            "recall plateaus and stage-2 latency grows. Outputs "
            "RERANK_topn_sweep.md, .csv, and .png to bench/results/."
        ),
    )
    p.add_argument(
        "--dataset", type=str, default="SPECTER2",
        help="text-bearing dataset (default: SPECTER2)",
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
        "--top-n", action="append", default=None,
        help=(
            "top_n values to sweep. May be passed repeatedly or "
            "comma-separated. Default: "
            f"{','.join(str(t) for t in DEFAULT_TOPN_LADDER)}."
        ),
    )
    p.add_argument(
        "--k-eval", type=int, default=DEFAULT_K_EVAL,
        help=f"R@K cutoff (default: {DEFAULT_K_EVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"quantizer rotation seed (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--cross-encoder-model", action="append", default=None,
        help=(
            "HF Hub model id for a cross-encoder. May be passed multiple "
            f"times — each becomes a stage-2b series. Default: "
            f"{DEFAULT_CROSS_ENCODER}."
        ),
    )
    p.add_argument(
        "--no-cross-encoder", action="store_true",
        help=(
            "skip stage-2b entirely (recall + latency for stage-1 and "
            "stage-2a only). Useful for fast plateau probes that don't "
            "need a model download."
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
        "--out-dir", type=str, default=None,
        help="output directory (default: bench/results/)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    spec = datasets.dataset_spec(args.dataset)  # validates name
    if not spec.has_texts:
        sys.stderr.write(
            f"error: dataset {args.dataset!r} has no registered texts cache; "
            f"the cross-encoder rerank experiment needs source text.\n"
        )
        return 2

    top_ns = _parse_top_ns(args.top_n)

    sys.stderr.write(f"[load] {args.dataset}: embeddings + texts\n")
    emb, einfo = datasets.load_dataset(args.dataset, n=args.n)
    texts, _ = datasets.load_texts(args.dataset, n=einfo["n"])

    sys.stderr.write(
        f"[run]  n={einfo['n']} d={einfo['dim']} "
        f"queries={args.queries} top_ns={list(top_ns)} k_eval={args.k_eval}\n"
    )

    if args.no_cross_encoder:
        ces: list = [_NullReranker()]
    else:
        model_ids = (
            args.cross_encoder_model
            if args.cross_encoder_model
            else [DEFAULT_CROSS_ENCODER]
        )
        ces = []
        for mid in model_ids:
            sys.stderr.write(f"[ce]   {mid} (preparing ONNX session)\n")
            ce = CrossEncoderReranker(
                model_id=mid,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            ce.prepare()
            ces.append(ce)

    def _progress(top_n, idx, total):
        sys.stderr.write(
            f"[step] {idx + 1}/{total}: top_n={top_n}\n"
        )

    results = run_topn_sweep(
        emb=emb, texts=texts,
        top_ns=top_ns, n_queries=args.queries,
        k_eval=args.k_eval, seed=args.seed,
        cross_encoders=ces, progress=_progress,
    )

    out_dir = _resolve_results_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rerank_topn_sweep.csv"
    png_path = out_dir / "rerank_topn_sweep.png"
    md_path = out_dir / "RERANK_topn_sweep.md"

    write_topn_sweep_csv(results, csv_path)
    sys.stderr.write(f"\nwrote {csv_path}\n")

    plot_topn_sweep(results, path=png_path)
    sys.stderr.write(f"wrote {png_path}\n")

    md = format_topn_sweep_md(
        results=results, dataset=args.dataset, seed=args.seed,
        plot_path="rerank_topn_sweep.png",
        csv_path="rerank_topn_sweep.csv",
    )
    md_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"wrote {md_path}\n")

    for r in results:
        ce_recalls = "  ".join(
            f"{_short_model_id(x['model_id'])}={x['recall_at_k']:.3f}"
            for x in r["stage2b_results"]
        )
        sys.stderr.write(
            f"       top_n={r['top_n']}  "
            f"s1={r['stage1_recall_at_k']:.3f}  "
            f"s2a={r['stage2a_recall_at_k']:.3f}  "
            f"{ce_recalls}\n"
        )
    return 0


class _NullReranker:
    """Placeholder used by ``--no-cross-encoder``: returns the first k of
    the candidate set without any work. Latency is reported but
    near-zero; the row exists so the CSV / plot keep a uniform schema."""

    model_id = "no-cross-encoder"

    def prepare(self):
        return self

    def rerank(self, *, query_text, candidate_idx, candidate_texts, k):
        return candidate_idx[:k]


if __name__ == "__main__":
    raise SystemExit(main())
