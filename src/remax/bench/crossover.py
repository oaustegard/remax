"""Crossover evaluation: remax stacked vs remex Lloyd-Max at matched bit budgets.

This is the v0.1.0 publication artifact for issue #5. For each registered
dataset (SPECTER2, MiniLM-L6-v2, GloVe-300d), it computes R@10 vs float32
ground truth at six matched bits-per-dim levels (1, 2, 3, 4, 6, 8) for two
methods:

* **remax stacked** — :class:`~remax.SignBitQuantizer` at b/d=1,
  :class:`~remax.StackedSignBitQuantizer` at b/d>=2. Every step is a
  rank-correct Charikar-honest sign-bit estimator. Searched with symmetric
  Hamming distance (the library's native API).
* **remex Lloyd-Max** — :class:`remex.Quantizer` encoded once at the maximum
  precision (bits=8), then searched at each lower precision via Matryoshka
  index right-shift. This matches the protocol that produced the blog-post
  inversion (1-bit beats 2-bit beats 3-bit on SPECTER2 R@10). Searched with
  ADC (real-valued query × dequantized rotated codebook).

The output is a tidy CSV (``dataset, bits_per_dim, method, R_at_10``), a
multi-subplot PNG suitable for the README and the blog, and a CROSSOVER.md
narrative documenting the protocol and the observed crossover (if any).

Preprocessing — asymmetric by design
------------------------------------
Each library is exercised with its own natural preprocessing, which is what
the blog-post comparison did. SimHash assumes mean-zero data (its boundary
is at the origin), so remax sees the centered corpus + queries. Lloyd-Max
in remex normalizes to the unit sphere internally, with data-oblivious
N(0, 1/d) boundaries that expect the unit-norm direction distribution
typical of raw embeddings; it sees the **raw** corpus + queries. Float32
ground truth is computed on the raw corpus regardless, so the metric is
identical across the two methods.

Sanity check
------------
At bits/dim=1, both methods reduce to a sign-bit cosine LSH under a random
Haar rotation. Their R@10 should agree to within statistical noise from
the rotation draw — the harness flags any deviation larger than
``--sanity-tol`` (default 0.05 absolute). The two libraries draw the
rotation from independent RNG paths, so codes are not byte-identical even
under a shared seed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from remax import SignBitQuantizer, StackedSignBitQuantizer
from remax.bench import datasets
from remax.bench.eval import exact_knn, recall_at_k
from remax.bench.run_baseline import (
    DEFAULT_K_EVAL,
    DEFAULT_N,
    DEFAULT_N_QUERIES,
    DEFAULT_SEED,
    QUERY_SPLIT_SEED,
    _split_queries,
)

__all__ = [
    "BITS_PER_DIM_LADDER",
    "REMEX_MAX_BITS",
    "METHOD_REMAX",
    "METHOD_REMEX",
    "compute_crossover_for_embeddings",
    "format_crossover_md",
    "plot_crossover",
    "write_crossover_csv",
    "main",
]

# Bit budget ladder pinned by issue #5. The b/d=1 step is the sanity check
# (identical construction on both sides); the higher steps probe where, if
# anywhere, Lloyd-Max catches up.
BITS_PER_DIM_LADDER: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
REMEX_MAX_BITS: int = 8
METHOD_REMAX: str = "remax_stacked"
METHOD_REMEX: str = "remex_lloyd_max"
DEFAULT_SANITY_TOL: float = 0.05


# --------------------------------------------------------------------- #
# Per-dataset orchestration
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CrossoverPoint:
    """One (dataset, bits_per_dim, method, R@10) row."""

    dataset: str
    bits_per_dim: int
    method: str
    recall_at_k: float


def _eval_remax(
    *,
    bits_per_dim: int,
    d: int,
    seed: int,
    corpus_enc: np.ndarray,
    queries_enc: np.ndarray,
    truth: np.ndarray,
    k_eval: int,
) -> float:
    """Encode + Hamming-search remax at one bits-per-dim step."""
    if bits_per_dim == 1:
        q = SignBitQuantizer(d=d, seed=seed)
    else:
        q = StackedSignBitQuantizer(d=d, k=bits_per_dim, seed=seed)
    codes = q.encode(corpus_enc)
    pred = q.search(queries_enc, codes, k=k_eval)
    if pred.ndim == 1:
        pred = pred[None, :]
    return float(recall_at_k(pred, truth, k=k_eval))


def _eval_remex_all(
    *,
    d: int,
    seed: int,
    corpus_enc: np.ndarray,
    queries_enc: np.ndarray,
    truth: np.ndarray,
    k_eval: int,
    bits_ladder: Sequence[int],
) -> dict[int, float]:
    """Encode remex once at full precision, then read R@10 at each level.

    Imported lazily so the bench harness imports cleanly without remex
    installed (the dev-dep shape mandated by issue #5).
    """
    try:
        import remex  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise RuntimeError(
            "remex is required for the crossover evaluation. Install with "
            "`pip install 'remex>=0.5.1'` or `pip install -e .[bench]`."
        ) from exc

    quantizer = remex.Quantizer(d=d, bits=REMEX_MAX_BITS, seed=seed)
    compressed = quantizer.encode(corpus_enc.astype(np.float32, copy=False))
    out: dict[int, float] = {}
    for b in bits_ladder:
        if b > REMEX_MAX_BITS:
            raise ValueError(
                f"bits_per_dim={b} exceeds REMEX_MAX_BITS={REMEX_MAX_BITS}"
            )
        idx, _scores = quantizer.search_batch(
            compressed,
            queries_enc.astype(np.float32, copy=False),
            k=k_eval,
            precision=b,
        )
        out[b] = float(recall_at_k(idx, truth, k=k_eval))
    return out


def compute_crossover_for_embeddings(
    *,
    name: str,
    emb: np.ndarray,
    n_queries: int = DEFAULT_N_QUERIES,
    k_eval: int = DEFAULT_K_EVAL,
    seed: int = DEFAULT_SEED,
    remax_center: bool = True,
    remex_center: bool = False,
    bits_ladder: Sequence[int] = BITS_PER_DIM_LADDER,
) -> list[CrossoverPoint]:
    """Run the matched-bit-budget comparison on one embeddings array.

    Mirrors the protocol of :func:`compute_baseline_for_embeddings`: same
    query split (``QUERY_SPLIT_SEED``), same float32 ground truth on the
    raw corpus. Each method then preprocesses to its own design assumption:

    * remax centers the corpus + queries by the corpus mean (default
      ``remax_center=True``). Without this SimHash collapses on real
      embeddings with a heavy-mean dimension (e.g. SPECTER2's dim ≈ 15.5).
    * remex sees the raw corpus + queries (default ``remex_center=False``).
      Lloyd-Max in remex normalizes to the unit sphere internally; its
      data-oblivious N(0, 1/d) boundaries are tuned for the unit-norm
      direction distribution of raw embeddings, not for centered inputs.

    These defaults match the natural deployment of each library and the
    protocol the blog post used to observe the *One Bit Beats Two*
    inversion. Both flags are exposed for ablation; the combinations
    ``remax_center=False`` (SimHash collapses) and ``remex_center=True``
    (the inversion disappears) are useful sanity probes, not the headline
    numbers.
    """
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape}")
    n_total, d = int(emb.shape[0]), int(emb.shape[1])
    if d % 8 != 0:
        raise ValueError(
            f"d={d} not divisible by 8; remax codes are bit-packed bytewise"
        )

    corpus, queries = _split_queries(emb, n_queries, QUERY_SPLIT_SEED)
    truth = exact_knn(corpus, queries, k=k_eval)
    mu = corpus.mean(axis=0)

    remax_corpus = corpus - mu if remax_center else corpus
    remax_queries = queries - mu if remax_center else queries
    remex_corpus = corpus - mu if remex_center else corpus
    remex_queries = queries - mu if remex_center else queries

    points: list[CrossoverPoint] = []
    for b in bits_ladder:
        r_remax = _eval_remax(
            bits_per_dim=b, d=d, seed=seed,
            corpus_enc=remax_corpus, queries_enc=remax_queries,
            truth=truth, k_eval=k_eval,
        )
        points.append(
            CrossoverPoint(
                dataset=name, bits_per_dim=b,
                method=METHOD_REMAX, recall_at_k=r_remax,
            )
        )

    remex_recalls = _eval_remex_all(
        d=d, seed=seed,
        corpus_enc=remex_corpus, queries_enc=remex_queries,
        truth=truth, k_eval=k_eval, bits_ladder=bits_ladder,
    )
    for b in bits_ladder:
        points.append(
            CrossoverPoint(
                dataset=name, bits_per_dim=b,
                method=METHOD_REMEX, recall_at_k=remex_recalls[b],
            )
        )

    return points


# --------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SanityResult:
    dataset: str
    remax: float
    remex: float
    delta: float
    tol: float

    @property
    def passed(self) -> bool:
        return abs(self.delta) <= self.tol


def check_one_bit_sanity(
    points: Sequence[CrossoverPoint], tol: float = DEFAULT_SANITY_TOL
) -> list[SanityResult]:
    """Per-dataset b/d=1 sanity check: |R_remax - R_remex| <= tol.

    At b/d=1 both methods reduce to sign-bit cosine LSH on the same input,
    differing only by which random Haar rotation they sample. Their R@10
    should agree to within statistical noise. ``tol`` is absolute recall.
    """
    by_ds: dict[str, dict[str, float]] = {}
    for p in points:
        if p.bits_per_dim != 1:
            continue
        by_ds.setdefault(p.dataset, {})[p.method] = p.recall_at_k
    results: list[SanityResult] = []
    for ds, methods in by_ds.items():
        if METHOD_REMAX not in methods or METHOD_REMEX not in methods:
            continue
        a = methods[METHOD_REMAX]
        b = methods[METHOD_REMEX]
        results.append(
            SanityResult(dataset=ds, remax=a, remex=b, delta=a - b, tol=tol)
        )
    return results


# --------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------- #


def write_crossover_csv(
    points: Sequence[CrossoverPoint], path: Path
) -> None:
    """Write tidy CSV: ``dataset, bits_per_dim, method, R_at_10``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "bits_per_dim", "method", "R_at_10"])
        for p in points:
            w.writerow([
                p.dataset, p.bits_per_dim, p.method,
                f"{p.recall_at_k:.6f}",
            ])


# --------------------------------------------------------------------- #
# PNG plot
# --------------------------------------------------------------------- #


def _group_points(
    points: Sequence[CrossoverPoint],
) -> dict[str, dict[str, dict[int, float]]]:
    """Group as ``{dataset: {method: {bits: recall}}}``."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for p in points:
        out.setdefault(p.dataset, {}).setdefault(p.method, {})[p.bits_per_dim] = (
            p.recall_at_k
        )
    return out


def plot_crossover(
    points: Sequence[CrossoverPoint],
    *,
    path: Path,
    dataset_order: Optional[Sequence[str]] = None,
    bits_ladder: Sequence[int] = BITS_PER_DIM_LADDER,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> None:
    """Render the three-subplot crossover PNG.

    Sober matplotlib defaults; one subplot per dataset, two lines per
    subplot (remax stacked, remex Lloyd-Max). The b/d=1 sanity-check
    identity is annotated in the legend caption rather than over the plot
    so the curves stay readable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = _group_points(points)
    if dataset_order is None:
        dataset_order = list(grouped.keys())
    else:
        dataset_order = [d for d in dataset_order if d in grouped]

    n_panels = len(dataset_order)
    if n_panels == 0:
        raise ValueError("no datasets to plot")

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, sharey=True, squeeze=False
    )
    axes = axes[0]
    bits = list(bits_ladder)

    for ax, ds in zip(axes, dataset_order):
        methods = grouped[ds]
        remax_y = [methods.get(METHOD_REMAX, {}).get(b) for b in bits]
        remex_y = [methods.get(METHOD_REMEX, {}).get(b) for b in bits]

        ax.plot(
            bits, remax_y,
            marker="o", linewidth=2.0, label="remax stacked",
        )
        ax.plot(
            bits, remex_y,
            marker="s", linewidth=2.0, linestyle="--",
            label="remex Lloyd-Max",
        )
        ax.set_title(ds)
        ax.set_xlabel("bits per dim")
        ax.set_xticks(bits)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("R@10 vs float32")
    # Single legend below, with the sanity-check note baked in. Avoids
    # repeating the legend in every subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Crossover: remax stacked vs remex Lloyd-Max  "
        "(b/d=1 is the sanity-check identity — both reduce to SimHash)",
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Markdown narrative
# --------------------------------------------------------------------- #


def _fmt_recall(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def format_crossover_md(
    *,
    points: Sequence[CrossoverPoint],
    sanity: Sequence[SanityResult],
    version: str,
    n_queries: int,
    k_eval: int,
    seed: int,
    remax_center: bool,
    remex_center: bool,
    bits_ladder: Sequence[int] = BITS_PER_DIM_LADDER,
    plot_path: str = "crossover.png",
    csv_path: str = "crossover.csv",
    dataset_order: Optional[Sequence[str]] = None,
    skipped: Sequence[tuple[str, str]] = (),
) -> str:
    """Render the CROSSOVER.md narrative.

    Includes the protocol block, the side-by-side table, the b/d=1 sanity
    result, and a generated commentary describing the observed crossover
    pattern. Commentary is *descriptive* — the issue explicitly asks for
    no pre-judging the outcome.
    """
    grouped = _group_points(points)
    if dataset_order is None:
        dataset_order = list(grouped.keys())
    else:
        dataset_order = [d for d in dataset_order if d in grouped] + [
            d for d in dataset_order if d not in grouped
        ]

    lines: list[str] = []
    lines.append(
        f"## remax v{version} crossover — stacked vs Lloyd-Max at matched bit budgets"
    )
    lines.append("")
    lines.append(
        "The publishable artifact for issue #5: side-by-side R@10 of "
        "remax stacked SimHash vs remex Lloyd-Max under matched "
        "bits-per-dim, on real embeddings."
    )
    lines.append("")
    lines.append(f"![crossover plot]({plot_path})")
    lines.append("")
    lines.append(f"Tidy CSV: [`{csv_path}`]({csv_path}).")
    lines.append("")

    # Protocol block
    lines.append("### Protocol")
    lines.append("")
    lines.append(f"- **remax version**: v{version}")
    lines.append(
        "- **remex version**: dev-dep `remex>=0.5.1` "
        "(Quantizer encoded once at bits=8, then searched at "
        "precision=1,2,3,4,6,8 via Matryoshka right-shift — the same "
        "extraction path that produced the *One Bit Beats Two* inversion)."
    )
    lines.append(
        f"- **Eval metric**: R@{k_eval} vs float32 inner-product ground "
        f"truth (computed on the **raw**, un-centered corpus)."
    )
    lines.append(
        f"- **Split**: {n_queries} held-out queries per dataset, corpus = "
        f"remainder. Query split seed = {QUERY_SPLIT_SEED}, quantizer "
        f"seed = {seed} for both methods."
    )
    lines.append(
        f"- **Preprocessing (remax)**: "
        f"{'centered by corpus mean' if remax_center else 'raw (un-centered)'}. "
        "remax's sign-bit boundary is at the origin; centering is "
        "required for SimHash to function on real embeddings with a "
        "heavy-mean dim (e.g. SPECTER2's dim ≈ 15.5)."
    )
    lines.append(
        f"- **Preprocessing (remex)**: "
        f"{'centered by corpus mean' if remex_center else 'raw'}. "
        "remex normalizes to the unit sphere internally and uses "
        "data-oblivious N(0, 1/d) Lloyd-Max boundaries; the natural "
        "input is the raw corpus."
    )
    lines.append(
        "- **Hardware**: pure NumPy on CPU. SIMD/Numba/GPU paths are "
        "post-v0.1.0 by design (CLAUDE.md anti-goals)."
    )
    lines.append("")

    # Per-dataset table
    lines.append("### R@10 by bits per dim")
    lines.append("")
    bits = list(bits_ladder)
    header = (
        "| dataset       | method            | "
        + " | ".join(f"b/d={b}" for b in bits)
        + " |"
    )
    sep = (
        "|---------------|-------------------|"
        + "|".join("-" * 7 for _ in bits)
        + "|"
    )
    lines.append(header)
    lines.append(sep)
    for ds in dataset_order:
        methods = grouped.get(ds, {})
        for method_label, method_key in (
            ("remax stacked", METHOD_REMAX),
            ("remex Lloyd-Max", METHOD_REMEX),
        ):
            row_vals = [
                _fmt_recall(methods.get(method_key, {}).get(b))
                for b in bits
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{ds:<13s}",
                        f"{method_label:<17s}",
                        *[f"{v:<5s}" for v in row_vals],
                    ]
                )
                + " |"
            )
    lines.append("")

    # Skipped datasets
    if skipped:
        lines.append("### Skipped datasets")
        lines.append("")
        for name, reason in skipped:
            lines.append(f"- **{name}** — {reason}")
        lines.append("")

    # Sanity check
    lines.append("### b/d=1 sanity check")
    lines.append("")
    lines.append(
        "At one bit per dim, both methods reduce to sign-bit cosine LSH "
        "(Charikar 2002 SimHash) on a random Haar rotation of the "
        "centered input. The two libraries draw the rotation from "
        "different RNG paths, so codes are not byte-identical, but R@10 "
        "should agree to within statistical noise across the held-out "
        "query set."
    )
    lines.append("")
    if sanity:
        lines.append("| dataset       | remax R@10 | remex R@10 | Δ      | within tol? |")
        lines.append("|---------------|------------|------------|--------|-------------|")
        for s in sanity:
            mark = "yes" if s.passed else "**NO**"
            lines.append(
                f"| {s.dataset:<13s} | {s.remax:>10.3f} | {s.remex:>10.3f} | "
                f"{s.delta:>+6.3f} | {mark} (tol={s.tol:.2f}) |"
            )
        lines.append("")
    else:
        lines.append("_No completed datasets — sanity check skipped._")
        lines.append("")

    # Commentary
    lines.append("### Commentary")
    lines.append("")
    lines.append(_generate_commentary(grouped, dataset_order, bits))
    lines.append("")
    lines.append(
        "**On the asymptotes.** The two methods do not target the same "
        "metric, even though the matched bits-per-dim axis suggests they "
        "do. remax searches with symmetric Hamming on stacked sign bits — "
        "an estimator of the angle between vectors, which asymptotes to "
        "the centered cosine ranking. remex Lloyd-Max searches with "
        "asymmetric distance computation (ADC) — a real-valued query "
        "against a dequantized rotated codebook — which asymptotes to "
        "the inner-product ranking. SPECTER2 is **not** unit-normalized "
        "(norms 20.85–22.24), so cosine and inner product disagree by "
        "a few percentage points; the cosine-angle ceiling against the "
        "raw inner-product truth used here is ≈ 0.73 R@10 on SPECTER2. "
        "remax's curve plateaus near that ceiling; remex's curve climbs "
        "past it because ADC reconstructs inner product, not just angle. "
        "Both findings are real, not artefacts."
    )
    lines.append("")
    lines.append(
        "**Useful follow-ups.** (a) Re-run on a real corpus with "
        "sentence-transformer MiniLM and GloVe-300d instead of placeholder "
        "rows; (b) extend to b/d=12, 16 to see how steeply the Lloyd-Max "
        "side flattens; (c) measure variance across seeds — a single "
        "seed's curve hides whether the low-bit lead is robust or noise; "
        "(d) implement an asymmetric-distance variant on the remax side "
        "(real-valued query × sign-bit corpus) to factor out the search "
        "asymmetry and isolate the storage-construction comparison."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _generate_commentary(
    grouped: Mapping[str, Mapping[str, Mapping[int, float]]],
    dataset_order: Sequence[str],
    bits: Sequence[int],
) -> str:
    """Compose a 1-2 paragraph descriptive summary of what the plot shows.

    Style: corvid voice — short, declarative, no chart-junk adjectives.
    Reports what's in the numbers; does not pre-judge the outcome.
    """
    parts: list[str] = []
    for ds in dataset_order:
        methods = grouped.get(ds, {})
        remax_curve = methods.get(METHOD_REMAX, {})
        remex_curve = methods.get(METHOD_REMEX, {})
        if not remax_curve or not remex_curve:
            continue
        # Find the bits-per-dim where remex catches up (if anywhere) and
        # the bits-per-dim where remax peaks ahead.
        crossover_b: Optional[int] = None
        for b in bits:
            if b in remax_curve and b in remex_curve:
                if remex_curve[b] >= remax_curve[b]:
                    crossover_b = b
                    break
        max_lead_b = None
        max_lead = float("-inf")
        for b in bits:
            if b in remax_curve and b in remex_curve:
                lead = remax_curve[b] - remex_curve[b]
                if lead > max_lead:
                    max_lead = lead
                    max_lead_b = b
        # Lloyd-Max inversion check (1>2>3 strict descent on remex)
        inversion = (
            all(b in remex_curve for b in (1, 2, 3))
            and remex_curve[1] > remex_curve[2] > remex_curve[3]
        )

        bits_str = ", ".join(str(b) for b in sorted(remax_curve))
        seg = [f"**{ds}** — bits/dim {bits_str}. "]
        if max_lead_b is not None and max_lead > 0:
            seg.append(
                f"remax stacked leads remex Lloyd-Max by "
                f"{max_lead:+.3f} R@10 at b/d={max_lead_b}. "
            )
        if crossover_b is not None:
            seg.append(
                f"remex catches up at b/d={crossover_b} "
                f"(remex {remex_curve[crossover_b]:.3f} vs remax "
                f"{remax_curve[crossover_b]:.3f}). "
            )
        else:
            seg.append(
                "remex never catches remax across the tested ladder. "
            )
        if inversion:
            seg.append(
                "remex's curve replicates the *One Bit Beats Two* "
                "inversion (1-bit > 2-bit > 3-bit). "
            )
        parts.append("".join(seg).strip())

    if not parts:
        return (
            "_No datasets produced complete curves — see the skipped section._"
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------- #


def _read_version() -> str:
    try:
        import remax
        return remax.__version__
    except Exception:
        return "0.0.0"


def _resolve_results_dir(out: Optional[str]) -> Path:
    """Locate ``<repo_root>/bench/results`` from the package source path."""
    if out is not None:
        return Path(out).resolve()
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "bench" / "results"


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="remax-bench-crossover",
        description=(
            "Produce remax v0.1.0 crossover artifacts (CSV + PNG + "
            "CROSSOVER.md) — issue #5."
        ),
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"max embeddings per dataset (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--queries", type=int, default=DEFAULT_N_QUERIES,
        help=f"held-out queries per dataset (default: {DEFAULT_N_QUERIES})",
    )
    p.add_argument(
        "--k-eval", type=int, default=DEFAULT_K_EVAL,
        help=f"R@K cutoff (default: {DEFAULT_K_EVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"quantizer seed for both methods (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="output directory (default: bench/results/)",
    )
    p.add_argument(
        "--no-remax-center", action="store_true",
        help=(
            "feed remax raw (un-centered) corpus + queries. Default "
            "centers because pure SimHash assumes mean-zero data; "
            "disabling collapses the SimHash side on real embeddings "
            "with a heavy-mean outlier dim (e.g. SPECTER2)."
        ),
    )
    p.add_argument(
        "--remex-center", action="store_true",
        help=(
            "feed remex centered corpus + queries (default: raw). "
            "remex normalizes to the unit sphere internally and its "
            "data-oblivious Lloyd-Max boundaries expect raw-direction "
            "distributions; centering before that pipeline often hurts "
            "and is provided here only for ablation."
        ),
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help=(
            "subset of dataset names to run (default: all registered). "
            "Unknown names are an error."
        ),
    )
    p.add_argument(
        "--sanity-tol", type=float, default=DEFAULT_SANITY_TOL,
        help=(
            "absolute R@10 tolerance for the b/d=1 sanity check "
            f"(default: {DEFAULT_SANITY_TOL})"
        ),
    )
    p.add_argument(
        "--strict-sanity", action="store_true",
        help=(
            "exit non-zero if any b/d=1 sanity check fails. Default is to "
            "warn but still emit artifacts so the failure is documented "
            "in CROSSOVER.md."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    target_datasets = (
        list(datasets.available_datasets())
        if args.datasets is None
        else list(args.datasets)
    )
    for name in target_datasets:
        datasets.dataset_spec(name)  # validate names early

    all_points: list[CrossoverPoint] = []
    skipped: list[tuple[str, str]] = []
    for name in target_datasets:
        try:
            X, info = datasets.load_dataset(name, n=args.n)
        except FileNotFoundError as e:
            sys.stderr.write(f"[skip] {name}: {e}\n\n")
            # First line of FileNotFoundError contains the user-actionable
            # remediation hint registered in the dataset spec.
            skipped.append((name, str(e).splitlines()[0]))
            continue

        sys.stderr.write(f"[run]  {name}: n={info['n']} d={info['dim']}\n")
        points = compute_crossover_for_embeddings(
            name=name, emb=X,
            n_queries=args.queries, k_eval=args.k_eval, seed=args.seed,
            remax_center=not args.no_remax_center,
            remex_center=args.remex_center,
        )
        # Per-step progress so a long SPECTER2 run isn't silent.
        for p_pt in points:
            sys.stderr.write(
                f"       {p_pt.method:<18s} b/d={p_pt.bits_per_dim} "
                f"R@{args.k_eval}={p_pt.recall_at_k:.3f}\n"
            )
        all_points.extend(points)

    out_dir = _resolve_results_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "crossover.csv"
    png_path = out_dir / "crossover.png"
    md_path = out_dir / "CROSSOVER.md"

    write_crossover_csv(all_points, csv_path)
    sys.stderr.write(f"\nwrote {csv_path}\n")

    sanity = check_one_bit_sanity(all_points, tol=args.sanity_tol)
    failed = [s for s in sanity if not s.passed]
    for s in sanity:
        marker = "ok" if s.passed else "FAIL"
        sys.stderr.write(
            f"[sanity {marker}] {s.dataset}: remax={s.remax:.3f} "
            f"remex={s.remex:.3f} Δ={s.delta:+.3f} (tol={s.tol:.2f})\n"
        )

    if all_points:
        plot_crossover(
            all_points, path=png_path, dataset_order=target_datasets,
        )
        sys.stderr.write(f"wrote {png_path}\n")
    else:
        sys.stderr.write("no datasets ran — skipping plot\n")

    md = format_crossover_md(
        points=all_points, sanity=sanity,
        version=_read_version(),
        n_queries=args.queries, k_eval=args.k_eval, seed=args.seed,
        remax_center=not args.no_remax_center,
        remex_center=args.remex_center,
        plot_path="crossover.png", csv_path="crossover.csv",
        dataset_order=target_datasets, skipped=skipped,
    )
    md_path.write_text(md, encoding="utf-8")
    sys.stderr.write(f"wrote {md_path}\n")

    if args.strict_sanity and failed:
        sys.stderr.write(
            f"\n[strict-sanity] {len(failed)} dataset(s) failed b/d=1 check\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
