"""Master comparison plot: every Nemotron-3-Embed-1B compression method, one chart.

Reads the per-method CSVs produced by the other bench scripts and draws two
panels (SciFact nDCG@10, STS-B Spearman) of quality vs bytes-per-vector, with
all families overlaid:

  - full-precision references: BF16 full, NVFP4 full (model-weight quantized)
  - remax     : 1-bit sign codes + stacked (k=2,4) + 1-bit on MRL slices
  - remex     : Lloyd-Max scalar 2/3/4/8-bit  (the rank-broken middle)
  - int8      : scalar quantization
  - PQ        : product quantization (data-dependent)
  - float32   : Matryoshka truncation

Usage
-----
    python bench/plot_nemotron_master.py
    python bench/plot_nemotron_master.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(results: Path):
    """Return {'scifact': {family: [(bytes, y, err, label)]}, 'stsb': {...}}."""
    sci: dict = {k: [] for k in ("bf16", "nvfp4", "remax", "remex", "int8", "pq", "float")}
    sts: dict = {k: [] for k in sci}

    def rd(name):
        p = results / name
        return list(csv.DictReader(open(p))) if p.exists() else []

    # remax bit methods + float-truncation refs (multi-seed)
    for r in rd("nemotron_seeds.csv"):
        b = int(r["bytes_per_vec"])
        if r["dataset"] == "scifact":
            y, e = _f(r["ndcg10_mean"]), _f(r["ndcg10_std"]) or 0
        else:
            y, e = _f(r["spearman_mean"]), _f(r["spearman_std"]) or 0
        if y is None:
            continue
        fam = "remax" if r["method"].startswith("bit1") else "float"
        (sci if r["dataset"] == "scifact" else sts)[fam].append((b, y, e, r["method"]))

    # int8 + pq
    for r in rd("nemotron_baselines.csv"):
        b = int(r["bytes_per_vec"])
        fam = r["family"]
        if r["dataset"] == "scifact":
            y = _f(r["ndcg10"])
        else:
            y = _f(r["spearman"])
        if y is None:
            continue
        (sci if r["dataset"] == "scifact" else sts)[fam].append((b, y, 0, r["method"]))

    # remex Lloyd-Max ladder (multi-seed)
    for r in rd("nemotron_remex.csv"):
        b = int(round(float(r["bytes_per_vec"])))
        if r["dataset"] == "scifact":
            y, e = _f(r["ndcg10_mean"]), _f(r["ndcg10_std"]) or 0
        else:
            y, e = _f(r["spearman_mean"]), _f(r["spearman_std"]) or 0
        if y is None:
            continue
        (sci if r["dataset"] == "scifact" else sts)["remex"].append((b, y, e, r["method"]))

    # NVFP4 + BF16 full-precision references (from the composition eval)
    for r in rd("nemotron_nvfp4.csv"):
        if r["method"] != "full_f32":
            continue
        b = int(r["bytes_per_vec"])
        if r["dataset"] == "scifact":
            y = _f(r["ndcg10"])
        else:
            y = _f(r["spearman"])
        if y is None:
            continue
        fam = r["embset"]  # bf16 | nvfp4
        (sci if r["dataset"] == "scifact" else sts)[fam].append((b, y, 0, fam))
    return {"scifact": sci, "stsb": sts}


STYLE = {
    "remax": ("#d1495b", "o", "remax sign codes: 1-bit + stacked + MRL (data-oblivious)"),
    "remex": ("#7b2cbf", "v", "remex Lloyd-Max 2/3/4-bit (data-oblivious)"),
    "int8": ("#2e7d32", "^", "int8 scalar"),
    "pq": ("#1f6feb", "s", "product quant. (data-dependent)"),
    "float": ("#9a9a9a", "D", "float32 Matryoshka truncation"),
}


def draw(data, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    panels = [
        ("scifact", "nDCG@10  (retrieval quality)",
         "SciFact retrieval\nRanking quality vs. index size"),
        ("stsb", "Spearman ρ  (similarity quality)",
         "STS-B similarity\nPairwise-similarity quality vs. index size"),
    ]
    for ax, (ds, ylab, title) in zip(axes, panels):
        fam = data[ds]
        for key in ("float", "int8", "pq", "remex", "remax"):
            pts = sorted(fam.get(key, []))
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
            c, mk, lab = STYLE[key]
            z = 6 if key == "remax" else (5 if key == "remex" else 3)
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=c, label=lab, lw=2.0,
                        ms=8, capsize=3, alpha=0.95, zorder=z)
        # full-precision reference points (BF16 ≈ NVFP4, so draw BF16 larger and
        # behind, NVFP4 smaller and on top, so both are visible at ~8192 B).
        for key, col, mk, sz, zo, lab in [
            ("bf16", "#000000", "*", 320, 8, "BF16 full (uncompressed)"),
            ("nvfp4", "#e07a00", "P", 110, 9, "NVFP4 full (NVIDIA model-quant)"),
        ]:
            for b, y, _, _ in fam.get(key, []):
                ax.scatter([b], [y], color=col, marker=mk, s=sz, zorder=zo,
                           edgecolor="white", linewidth=0.7, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bytes per vector  —  smaller index ← → larger index (log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(256, ls="--", color="#d1495b", alpha=0.2, lw=1.2)
    # de-duplicate legend labels
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    axes[0].legend(seen.values(), seen.keys(), fontsize=8.2, loc="lower right", framealpha=0.95)
    fig.suptitle(
        "Compressing NVIDIA Nemotron-3-Embed-1B embeddings: every method, matched byte budgets\n"
        "Model-weight quantization (NVFP4) and embedding quantization (remax/remex/int8/PQ) on one axis — higher is better",
        fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


def selftest():
    import tempfile
    print("[plot_master] selftest ...")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "nemotron_seeds.csv").write_text(
            "dataset,method,dims,k_stack,bytes_per_vec,n_seeds,r10_mean,r10_std,ndcg10_mean,ndcg10_std,spearman_mean,spearman_std\n"
            "scifact,bit1_2048,2048,1,256,5,0.75,0.01,0.82,0.01,,\n"
            "scifact,f32_2048,2048,1,8192,1,1.0,0,0.84,0,,\n"
            "stsb,bit1_2048,2048,1,256,5,,,,,0.84,0.001\n")
        (d / "nemotron_baselines.csv").write_text(
            "dataset,method,family,dims_or_M,bytes_per_vec,compression_x,r10_vs_float,ndcg10,spearman\n"
            "scifact,int8_2048,int8,2048,2048,4,0.99,0.84,\n"
            "scifact,pq_m256,pq,256,256,32,0.75,0.81,\n"
            "stsb,int8_2048,int8,2048,2048,4,,,0.85\n")
        (d / "nemotron_remex.csv").write_text(
            "dataset,method,bits,bytes_per_vec,compression_x,r10_mean,r10_std,ndcg10_mean,ndcg10_std,spearman_mean,spearman_std\n"
            "scifact,remex_2bit,2,516,15.9,0.7,0.01,0.79,0.01,,\n"
            "stsb,remex_2bit,2,516,15.9,,,,,0.82,0.01\n")
        (d / "nemotron_nvfp4.csv").write_text(
            "dataset,embset,method,bytes_per_vec,r10_vs_bf16,ndcg10,spearman\n"
            "scifact,bf16,full_f32,8192,1.0,0.84,\n"
            "scifact,nvfp4,full_f32,8192,0.95,0.84,\n"
            "stsb,nvfp4,full_f32,8192,,,0.848\n")
        data = load(d)
        assert data["scifact"]["remex"], "remex points missing"
        assert data["scifact"]["nvfp4"], "nvfp4 ref missing"
        out = d / "master.png"
        draw(data, out)
        assert out.exists() and out.stat().st_size > 0
    print("PASS: master plot builds from synthetic CSVs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    draw(load(RESULTS), RESULTS / "nemotron_master.png")


if __name__ == "__main__":
    main()
