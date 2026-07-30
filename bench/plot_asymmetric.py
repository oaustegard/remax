"""Figure for the asymmetric-scoring result.

Left  — the gap between symmetric and asymmetric widens as the index shrinks.
Right — asymmetry vs buying the same accuracy with more bits, at matched bytes.

    python bench/plot_asymmetric.py --results bench/results/lfm25_asymmetric.json \
                                    --out bench/results
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

SYM = "#c0504d"
ASYM = "#4f81bd"
GREY = "#808080"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results/lfm25_asymmetric.json")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--label", default="LFM2.5-Embedding-350M")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.results).read_text())
    fp32 = data["fp32"]["ndcg@10"]
    rows = sorted(data["rows"], key=lambda r: r["bytes_per_vec"])
    stacked = sorted(data.get("stacked", []), key=lambda r: r["bytes_per_vec"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.suptitle(
        f"Keeping the query in float is free — {args.label}, BEIR SciFact",
        fontweight="bold", fontsize=13.5,
    )

    # ---- left: the gap widens as the index shrinks ----
    x = [r["bytes_per_vec"] for r in rows]
    ys = [r["symmetric"]["ndcg@10"] for r in rows]
    ya = [r["asymmetric"]["ndcg@10"] for r in rows]

    ax1.axhline(fp32, ls="--", color=GREY, lw=1.3,
                label=f"uncompressed fp32 = {fp32:.4f}")
    ax1.plot(x, ys, "o-", color=SYM, lw=2, ms=7,
             label="symmetric (query binarized too)")
    ax1.plot(x, ya, "v-", color=ASYM, lw=2, ms=7,
             label="asymmetric (query kept in float)")
    ax1.fill_between(x, ys, ya, color=ASYM, alpha=0.12)

    for xi, s, a in zip(x, ys, ya):
        ax1.annotate(f"+{a - s:.3f}", (xi, (s + a) / 2),
                     textcoords="offset points", xytext=(8, -3),
                     fontsize=9, color=ASYM, fontweight="bold")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in x])
    ax1.set_xlabel("bytes per vector  —  smaller index ←", fontsize=10)
    ax1.set_ylabel("nDCG@10", fontsize=10)
    ax1.set_title("The harder you compress, the more it buys",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # ---- right: asymmetry vs spending bytes ----
    # Dumbbell rather than bars: the differences here are a few thousandths, so
    # the axis has to be clipped to show them at all, and clipped bars lie about
    # ratios. A dot plot has no baseline to misread.
    if stacked:
        pos = list(range(len(stacked)))
        s_vals = [r["symmetric"]["ndcg@10"] for r in stacked]
        a_vals = [r["asymmetric"]["ndcg@10"] for r in stacked]

        ax2.axhline(fp32, ls="--", color=GREY, lw=1.3,
                    label=f"uncompressed fp32 = {fp32:.4f}")
        # The headline, drawn rather than annotated: asymmetric at the smallest
        # index sits above symmetric at every larger one.
        ax2.axhline(a_vals[0], ls=":", color=ASYM, lw=1.6)
        ax2.text(len(stacked) - 0.55, a_vals[0] + 0.0012,
                 f"asymmetric k=1 ({stacked[0]['bytes_per_vec']} B) = {a_vals[0]:.4f}",
                 fontsize=9, color=ASYM, fontweight="bold", ha="right")

        for p, s, a in zip(pos, s_vals, a_vals):
            ax2.plot([p, p], [s, a], "-", color=ASYM, lw=2.5, alpha=0.45,
                     zorder=1)
            ax2.plot(p, s, "o", color=SYM, ms=11, zorder=2)
            ax2.plot(p, a, "o", color=ASYM, ms=11, zorder=2)
            ax2.text(p + 0.10, s, f"{s:.4f}", fontsize=9, va="center",
                     color=SYM)
            ax2.text(p + 0.10, a, f"{a:.4f}", fontsize=9, va="center",
                     color=ASYM, fontweight="bold")

        ax2.plot([], [], "o", color=SYM, ms=9, label="symmetric")
        ax2.plot([], [], "o", color=ASYM, ms=9, label="asymmetric")

        ax2.set_xticks(pos)
        ax2.set_xticklabels(
            [f"k={r['k']}\n{r['bytes_per_vec']} B" for r in stacked], fontsize=9.5)
        ax2.set_xlim(-0.45, len(stacked) - 0.30)
        lo = min(min(s_vals), min(a_vals))
        hi = max(max(a_vals), fp32)
        ax2.set_ylim(lo - 0.006, hi + 0.006)
        ax2.set_ylabel("nDCG@10", fontsize=10)
        ax2.set_title("Asymmetry vs buying the same accuracy with more bits",
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9, loc="lower right", framealpha=0.95)
        ax2.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
    else:
        ax2.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "lfm25_asymmetric.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"wrote {dest}")

    print(f"\nfp32 baseline {fp32:.4f}")
    for r in rows:
        d = r["asymmetric"]["ndcg@10"] - r["symmetric"]["ndcg@10"]
        print(f"  {r['bytes_per_vec']:>4} B  sym {r['symmetric']['ndcg@10']:.4f}"
              f"  asym {r['asymmetric']['ndcg@10']:.4f}  {d:+.4f}")
    for r in stacked:
        d = r["asymmetric"]["ndcg@10"] - r["symmetric"]["ndcg@10"]
        print(f"  k={r['k']} ({r['bytes_per_vec']:>4} B)  "
              f"sym {r['symmetric']['ndcg@10']:.4f}  "
              f"asym {r['asymmetric']['ndcg@10']:.4f}  {d:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
