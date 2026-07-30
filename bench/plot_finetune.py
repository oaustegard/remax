"""Figure for the fine-tuning result: the proxy vs the task.

Left  — quantization error against retrieval quality across the four transforms.
        The point is the anti-correlation: the method with the lowest error has
        the worst nDCG.
Right — what an ordinary contrastive fine-tune did to the fp32/1-bit gap on
        held-out queries, with no quantization term in its objective.

    python bench/plot_finetune.py --out bench/results
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

FP32 = "#4f81bd"
BIN = "#c0504d"
GREY = "#808080"
BAD = "#8064a2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation", default="bench/results/lfm25_learned_rotation.json")
    ap.add_argument("--finetune", default="bench/results/lfm25_finetune_retrieval.json")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    rot = json.loads(pathlib.Path(args.rotation).read_text())
    ft = json.loads(pathlib.Path(args.finetune).read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.suptitle("Quantization error is a proxy, and optimizing it backfires",
                 fontweight="bold", fontsize=13.5)

    # ---- left: the anti-correlation ----
    pretty = {"haar": "random rotation", "itq": "ITQ (learned)",
              "ste": "STE / quantization-aware", "ste-orth": "STE, orthogonal"}
    # Three of the four cluster tightly in x and differ only in y, so the
    # offsets are set per method rather than by a rule -- otherwise the labels
    # sit on top of each other and the STE one runs off the axes.
    off = {"haar": (14, -16, "left"), "itq": (14, 8, "left"),
           "ste-orth": (14, 6, "left"), "ste": (-14, 14, "right")}
    for r in rot["rows"]:
        x = r["quantization_error"]
        y = r["symmetric"]["ndcg@10"]
        bad = r["method"] == "ste"
        dx, dy, ha = off.get(r["method"], (12, 10, "left"))
        ax1.scatter(x, y, s=190, zorder=3,
                    color=BAD if bad else FP32,
                    edgecolor="white", linewidth=1.5)
        ax1.annotate(pretty.get(r["method"], r["method"]), (x, y),
                     textcoords="offset points", xytext=(dx, dy), ha=ha,
                     fontsize=10, fontweight="bold" if bad else "normal",
                     color=BAD if bad else "black")
    ax1.margins(x=0.16, y=0.10)

    ax1.annotate(
        "lowest quantization error,\nworst retrieval — the map collapsed\n"
        "(orthogonality error 98.4)",
        xy=(rot["rows"][2]["quantization_error"],
            rot["rows"][2]["symmetric"]["ndcg@10"]),
        xytext=(0.34, 0.52), textcoords="axes fraction",
        fontsize=9.5, color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1.5))

    ax1.set_xlabel("quantization error  —  lower is 'better' by the proxy →",
                   fontsize=10)
    ax1.set_ylabel("nDCG@10  (what you actually want)", fontsize=10)
    ax1.set_title("Optimizing the proxy directly", fontsize=11, fontweight="bold")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.25)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- right: the fine-tune closing the gap ----
    b, f = ft["base"]["val"], ft["finetuned"]["val"]
    pos = [0, 1]
    fp = [b["fp32"]["ndcg@10"], f["fp32"]["ndcg@10"]]
    bn = [b["1bit_asym"]["ndcg@10"], f["1bit_asym"]["ndcg@10"]]

    for p, a, c in zip(pos, fp, bn):
        ax2.plot([p, p], [c, a], "-", color=GREY, lw=2.5, alpha=0.5, zorder=1)
        ax2.annotate(f"gap {a - c:.4f}", (p, (a + c) / 2),
                     textcoords="offset points", xytext=(14, -4),
                     fontsize=10, fontweight="bold", color=GREY)
    ax2.plot(pos, fp, "o-", color=FP32, ms=13, lw=2, label="fp32", zorder=2)
    ax2.plot(pos, bn, "o-", color=BIN, ms=13, lw=2,
             label="1-bit (128 B/vector)", zorder=2)
    for p, a, c in zip(pos, fp, bn):
        ax2.text(p, a + 0.006, f"{a:.4f}", ha="center", fontsize=9.5, color=FP32)
        ax2.text(p, c - 0.011, f"{c:.4f}", ha="center", fontsize=9.5, color=BIN)

    ax2.set_xticks(pos)
    ax2.set_xticklabels(["base model", "after fine-tuning\n(no quantization\nin the objective)"],
                        fontsize=10)
    ax2.set_xlim(-0.45, 1.55)
    lo, hi = min(bn) - 0.025, max(fp) + 0.022
    ax2.set_ylim(lo, hi)
    ax2.set_ylabel("nDCG@10  (held-out queries)", fontsize=10)
    ax2.set_title("Optimizing the task instead", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "lfm25_finetune.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"wrote {dest}")
    print(f"\n  base       fp32 {fp[0]:.4f}  1bit {bn[0]:.4f}  gap {fp[0]-bn[0]:.4f}")
    print(f"  fine-tuned fp32 {fp[1]:.4f}  1bit {bn[1]:.4f}  gap {fp[1]-bn[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
