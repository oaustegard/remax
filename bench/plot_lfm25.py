"""Plot the LFM2.5-Embedding-350M codec bake-off at matched byte budgets.

Consumes the JSON written by ``bench/eval_lfm25.py`` and answers the only
question that matters when you are sizing an index: *at a given bytes-per-vector
budget, which codec should I actually store?*

Two panels, one x-axis (bytes/vector, log2):

  left   nDCG@10   — retrieval quality against the real SciFact qrels
  right  agree@10  — fidelity to the uncompressed fp32 ranking

The two can disagree, and that disagreement is the interesting part: a codec can
hold its nDCG while reshuffling the top-10 (it keeps *judged* documents high but
swaps near-equivalent neighbours), or it can track fp32 faithfully while both
rank badly. Neither metric alone tells you that.

Also emits a matched-bytes markdown table in the NEMOTRON_MASTER.md house style
and prints the deploy answer (cheapest codec clearing 99/97/95% of fp32) to
stdout.

Usage
-----
::

    python bench/plot_lfm25.py --results bench/results/lfm25_scifact.json \
                               --out bench/results

Outputs
-------
``<out>/lfm25_pareto.png`` and ``<out>/LFM25_MATCHED_BYTES.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Canonical draw/report order. Families absent from the JSON are skipped, and
# any family the eval grows later still renders via _FALLBACK.
FAMILY_ORDER = ("fp32", "int8", "remex", "remax-unc", "remax")

STYLE = {
    "fp32": ("#9a9a9a", "D", "-", "fp32 (full + Matryoshka prefix truncation)"),
    "int8": ("#2e7d32", "^", "-", "int8 per-vector scalar quant"),
    "remex": ("#7b2cbf", "v", "-", "remex Lloyd-Max scalar (1/2/3/4/8-bit)"),
    "remax-unc": ("#e07a00", "X", "--", "remax 1-bit, uncentered (ablation)"),
    "remax": ("#d1495b", "o", "-", "remax centered SimHash (1-bit + stacked)"),
}
_FALLBACK = ("#1f6feb", "s", "-", None)

# Codecs collide at matched budgets by construction, so families drawn earlier
# get a larger marker and sit behind: overlapping points read as rings, not one
# blob. Annotations are offset per family for the same reason.
_OFFSETS = ((0, 9), (0, -15), (0, 20), (0, -26))

# Quality bars for the "what should I deploy" answer, as a fraction of the
# fp32-full nDCG@10.
BARS = (0.99, 0.97, 0.95)


def _f(x):
    """Float or None — the eval may omit a metric for a codec that errored."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def style_for(family: str):
    c, mk, ls, lab = STYLE.get(family, _FALLBACK)
    return c, mk, ls, (lab or family)


def load(path: Path) -> dict:
    """Read the eval JSON into {'rows': [...], 'geometry': {}, 'prediction': {}}."""
    blob = json.loads(Path(path).read_text())
    rows = []
    for r in blob.get("rows", []):
        b, nd = _f(r.get("bytes_per_vec")), _f(r.get("ndcg@10"))
        if b is None or b <= 0 or nd is None:
            continue  # a codec that errored out, or a malformed row
        rows.append(
            {
                "label": str(r.get("label", "?")),
                "family": str(r.get("family", "?")),
                "dim": _f(r.get("dim")),
                "bytes": int(round(b)),
                "ndcg": nd,
                "agree": _f(r.get("agree@10")),
                "r10": _f(r.get("r@10")),
                "r100": _f(r.get("r@100")),
                "secs": _f(r.get("secs")),
            }
        )
    return {
        "rows": rows,
        "geometry": blob.get("geometry") or {},
        "prediction": blob.get("prediction") or {},
    }


def fp32_baseline(rows: list[dict]) -> dict | None:
    """The uncompressed reference: the widest fp32 row, i.e. the most bytes."""
    fp = [r for r in rows if r["family"] == "fp32"]
    if not fp:
        return None
    return max(fp, key=lambda r: (r["bytes"], r["ndcg"]))


def short_label(row: dict, full_d) -> str:
    """Drop the redundant family prefix and the default-dim suffix."""
    lab = row["label"]
    for pre in (row["family"], row["family"].split("-")[0]):
        if pre and lab.lower().startswith(pre.lower()):
            lab = lab[len(pre):].strip()
            break
    if full_d is not None:
        lab = lab.replace(f"d={int(full_d)}", "").strip()
    return lab or row["label"]


def by_family(rows: list[dict]) -> dict:
    fams: dict = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    for v in fams.values():
        v.sort(key=lambda r: r["bytes"])
    return fams


def family_sequence(fams: dict) -> list[str]:
    known = [f for f in FAMILY_ORDER if f in fams]
    return known + sorted(f for f in fams if f not in FAMILY_ORDER)


def budgets(rows: list[dict]) -> dict:
    """{bytes_per_vec: [rows sorted by nDCG@10 desc]} — the matched-bytes view."""
    out: dict = {}
    for r in rows:
        out.setdefault(r["bytes"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: (-r["ndcg"], r["label"]))
    return dict(sorted(out.items(), reverse=True))


# ---------------------------------------------------------------- figures


def _panel(ax, fams, order, key, base, full_d, ylab, title, annotate=True):
    drawn = 0
    for fi, fam in enumerate(order):
        pts = [r for r in fams[fam] if r[key] is not None]
        if not pts:
            continue
        c, mk, ls, lab = style_for(fam)
        ax.plot(
            [p["bytes"] for p in pts], [p[key] for p in pts],
            marker=mk, ls=ls, color=c, label=lab, lw=2.0,
            ms=max(5.0, 10.0 - 0.9 * fi), mec="white", mew=0.7,
            alpha=0.95, zorder=3 + fi,
        )
        drawn += 1
        if not annotate:
            continue
        for p in pts:
            ax.annotate(
                short_label(p, full_d), (p["bytes"], p[key]),
                textcoords="offset points", xytext=_OFFSETS[fi % len(_OFFSETS)],
                fontsize=6.6, ha="center", color=c, zorder=20 + fi,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.65),
            )
    if base is not None and base.get(key) is not None:
        ax.axhline(
            base[key], ls="--", color="#333333", alpha=0.6, lw=1.3, zorder=2,
            label=f"fp32 full ({base['bytes']} B) = {base[key]:.4f}",
        )
    ax.set_xscale("log", base=2)
    ax.margins(x=0.09, y=0.14)  # headroom so the point annotations stay inside
    ax.set_xlabel("bytes per vector  —  smaller index ← → larger index (log scale)", fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return drawn


def draw(data: dict, out_png: Path, label: str) -> None:
    rows = data["rows"]
    fams = by_family(rows)
    order = family_sequence(fams)
    base = fp32_baseline(rows)
    full_d = base["dim"] if base else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), dpi=150)
    _panel(
        axes[0], fams, order, "ndcg", base, full_d,
        "nDCG@10  (retrieval quality vs. qrels)",
        "Which codec wins at which byte budget\nSciFact nDCG@10 vs. index size",
    )
    _panel(
        axes[1], fams, order, "agree", base, full_d,
        "agree@10  (top-10 overlap with fp32)",
        "…and which one preserves the fp32 ranking\nRank fidelity vs. index size",
    )
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    axes[0].legend(seen.values(), seen.keys(), fontsize=8.2, loc="lower right", framealpha=0.95)

    sub = (
        "nDCG@10 measures relevance; agree@10 measures fidelity to the uncompressed ranking. "
        "Where they disagree, the codec is swapping near-equivalent neighbours."
    )
    fig.suptitle(
        f"Compressing {label} embeddings: every codec, matched byte budgets\n{sub}",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------- markdown


def _pct(v, base):
    return None if (v is None or not base) else 100.0 * v / base


def _num(v, spec=".4f", dash="—"):
    """Format a maybe-missing metric for a markdown cell."""
    return dash if v is None else format(v, spec)


def write_markdown(data: dict, out_md: Path, label: str, src: Path) -> None:
    rows = data["rows"]
    base = fp32_baseline(rows)
    bnd = base["ndcg"] if base else None
    full_d = base["dim"] if base else None
    fams = by_family(rows)
    order = family_sequence(fams)
    buckets = budgets(rows)

    L = [
        f"# {label} — codec bake-off at matched byte budgets (BEIR SciFact)",
        "",
        f"Chart: `{out_md.with_name('lfm25_pareto.png').name}`. "
        f"Source: `{src.name}` (written by `bench/eval_lfm25.py`).",
        "",
    ]
    if base:
        L += [
            f"Uncompressed reference: **{base['label']}** at {base['bytes']} B/vec, "
            f"nDCG@10 **{base['ndcg']:.4f}**. Every percentage below is relative to it.",
            "",
        ]
    else:
        L += ["_No fp32 row in the results — percentages omitted._", ""]

    p = data.get("prediction") or {}
    if p.get("regime"):
        sr = _f(p.get("sigma_ratio"))
        sr_s = f" (post-rotation sigma ratio {sr:.3f})" if sr is not None else ""
        L += [
            f"Geometry called it **{p['regime']}**{sr_s} — predicted: {p.get('expectation', 'n/a')}",
            "",
        ]

    # --- matched-bytes matrix, NEMOTRON_MASTER.md style -------------------
    L += [
        "## Matched bytes — best codec per family at each budget",
        "",
        "Cell = best nDCG@10 that family reaches at that budget, with the codec that got it. "
        "**Bold** = winner of the budget.",
        "",
        "| bytes/vec | " + " | ".join(order) + " |",
        "|---|" + "|".join([":-:"] * len(order)) + "|",
    ]
    for b, group in buckets.items():
        best = group[0]
        cells = []
        for fam in order:
            here = [r for r in group if r["family"] == fam]
            if not here:
                cells.append("—")
                continue
            top = max(here, key=lambda r: r["ndcg"])
            txt = f"{top['ndcg']:.3f} ({short_label(top, full_d)})"
            cells.append(f"**{txt}**" if top is best else txt)
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L.append("")

    # --- per-budget detail ------------------------------------------------
    L += ["## Every codec at every budget", ""]
    for b, group in buckets.items():
        comp = f" — {full_d * 4 / b:.1f}x smaller than fp32-full" if full_d else ""
        L += [
            f"### {b} bytes/vector{comp}",
            "",
            "| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |",
            "|---|---|--:|--:|--:|--:|--:|",
        ]
        for i, r in enumerate(group):
            pc = _pct(r["ndcg"], bnd)
            name = f"**{r['label']}**" if i == 0 and len(group) > 1 else r["label"]
            nd = f"**{r['ndcg']:.4f}**" if i == 0 and len(group) > 1 else f"{r['ndcg']:.4f}"
            pcs = "—" if pc is None else f"{pc:.1f}%"
            L.append(
                f"| {name} | {r['family']} | {nd} | {pcs} | "
                f"{_num(r['r10'])} | {_num(r['r100'])} | {_num(r['agree'])} |"
            )
        L.append("")

    # --- deploy answer ----------------------------------------------------
    L += ["## What to deploy", ""]
    if bnd:
        L += [
            "Cheapest codec (fewest bytes/vector) still clearing each quality bar:",
            "",
            "| bar | codec | bytes/vec | compression | nDCG@10 | % of fp32 | agree@10 |",
            "|---|---|--:|--:|--:|--:|--:|",
        ]
        for bar in BARS:
            pick = cheapest_at_bar(rows, bnd, bar)
            if pick is None:
                L.append(f"| >= {bar:.0%} | _nothing qualifies_ | — | — | — | — | — |")
                continue
            comp = f"{full_d * 4 / pick['bytes']:.1f}x" if full_d else "—"
            L.append(
                f"| >= {bar:.0%} | {pick['label']} | {pick['bytes']} | {comp} | "
                f"{pick['ndcg']:.4f} | {_pct(pick['ndcg'], bnd):.1f}% | "
                f"{_num(pick['agree'])} |"
            )
    else:
        L.append("_No fp32 baseline — cannot rank against it._")
    L.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print(f"wrote {out_md}")


# ---------------------------------------------------------------- summary


def cheapest_at_bar(rows: list[dict], base_ndcg: float, bar: float):
    """Fewest bytes/vector still at >= bar x fp32 nDCG@10; ties go to higher nDCG."""
    ok = [r for r in rows if r["ndcg"] >= bar * base_ndcg]
    return min(ok, key=lambda r: (r["bytes"], -r["ndcg"])) if ok else None


def summarize(data: dict, label: str, src: Path) -> None:
    rows = data["rows"]
    if not rows:
        print("no usable rows in the results JSON", file=sys.stderr)
        return
    base = fp32_baseline(rows)
    bnd = base["ndcg"] if base else None
    full_d = base["dim"] if base else None

    n = len(rows)
    print(f"\n{label} — SciFact codec bake-off  ({n} codec{'s' if n != 1 else ''} from {src})")
    if base:
        print(f"fp32 baseline: {base['label']}  {base['bytes']} B/vec  nDCG@10 {base['ndcg']:.4f}")
    else:
        print("fp32 baseline: MISSING (no fp32 family in results)")
    p = data.get("prediction") or {}
    if p.get("regime"):
        print(f"geometry regime: {p['regime']} — predicted: {p.get('expectation', 'n/a')}")

    print("\nWinner at each byte budget")
    for b, group in budgets(rows).items():
        w = group[0]
        pc = _pct(w["ndcg"], bnd)
        pcs = f"  {pc:5.1f}% of fp32" if pc is not None else ""
        ag = f"  agree@10 {w['agree']:.4f}" if w["agree"] is not None else ""
        alt = f"  (over {len(group) - 1} other{'s' if len(group) > 2 else ''})" if len(group) > 1 else ""
        print(f"  {b:>6} B  {w['label']:<30} nDCG@10 {w['ndcg']:.4f}{pcs}{ag}{alt}")

    print("\nDeploy answer — cheapest codec clearing each quality bar")
    if not bnd:
        print("  n/a — no fp32 baseline to measure against")
        return
    for bar in BARS:
        pick = cheapest_at_bar(rows, bnd, bar)
        if pick is None:
            print(f"  >= {bar:.0%} of fp32 nDCG@10 (>= {bar * bnd:.4f})  nothing qualifies")
            continue
        comp = f"  {full_d * 4 / pick['bytes']:5.1f}x smaller" if full_d else ""
        ag = f"  agree@10 {pick['agree']:.4f}" if pick["agree"] is not None else ""
        print(
            f"  >= {bar:.0%} of fp32 nDCG@10 (>= {bar * bnd:.4f})  "
            f"{pick['label']:<30} {pick['bytes']:>6} B{comp}  "
            f"nDCG@10 {pick['ndcg']:.4f} ({_pct(pick['ndcg'], bnd):.1f}%){ag}"
        )


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("bench/results/lfm25_scifact.json"),
                    help="lfm25_scifact.json written by bench/eval_lfm25.py")
    ap.add_argument("--out", type=Path, default=Path("bench/results"),
                    help="output directory for the PNG and the markdown table")
    ap.add_argument("--label", default="LFM2.5-Embedding-350M")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: results not found: {args.results}", file=sys.stderr)
        return 1
    data = load(args.results)
    if not data["rows"]:
        print(f"ERROR: no usable rows in {args.results}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    draw(data, args.out / "lfm25_pareto.png", args.label)
    write_markdown(data, args.out / "LFM25_MATCHED_BYTES.md", args.label, args.results)
    summarize(data, args.label, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
