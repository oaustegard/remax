"""Evaluate remax across {adapter, matryoshka-dim, ladder-step}.

Inputs:  bench/.cache/JINA_V5_NANO/{task}.npy  shape (N, 768) float32, L2-normalized.

For each (task, dim):
  - Slice to first `dim` coordinates and re-normalize.
  - Hold out QUERY_SPLIT_SEED=99 permutation of N_QUERIES queries; rest = corpus.
  - Compute two ground truths:
      * truth_at_d   = top-k float32 IP on the dim-d sliced corpus
      * truth_at_768 = top-k float32 IP on the FULL 768-dim corpus
  - Evaluate remax ladder (1-bit, k=2, k=4, k=8) with centering.
    Recall vs truth_at_d  → "quantizer-only" recall (isolates remax quality)
    Recall vs truth_at_768 → "end-to-end" recall  (conflates matryoshka loss + quantizer loss)
  - Float32 R@K at dim d vs truth_at_768 is the matryoshka-only baseline.

Outputs:
  bench/results/JINA_V5_NANO.md     human-readable
  bench/results/jina_v5_nano.csv    raw numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from remax import SignBitQuantizer, StackedSignBitQuantizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "bench" / ".cache" / "JINA_V5_NANO"
RESULTS = REPO_ROOT / "bench" / "results"

TASKS = ("retrieval", "text-matching", "clustering", "classification")
DIMS = (32, 64, 128, 256, 512, 768)
LADDER_KS = (2, 4, 8)
N_QUERIES = 100
QUERY_SPLIT_SEED = 99      # mirrors remax bench convention
QUANTIZER_SEED = 42
K_EVALS = (10, 100)


def split(emb: np.ndarray, n_queries: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(emb.shape[0])
    return emb[perm[n_queries:]], emb[perm[:n_queries]]


def topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    # IP only — vectors are unit-norm, so IP ≡ cosine.
    scores = queries @ corpus.T
    # np.argpartition then sort the partition
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    # stable order by score (descending)
    row_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    # both shape (Q, >=k); compare top-k of each
    pred_k = pred[:, :k]
    truth_k = truth[:, :k]
    Q = pred_k.shape[0]
    hits = 0
    for q in range(Q):
        hits += len(set(pred_k[q].tolist()) & set(truth_k[q].tolist()))
    return hits / (Q * k)


def slice_renorm(emb: np.ndarray, d: int) -> np.ndarray:
    if d == emb.shape[1]:
        return emb.copy()
    out = emb[:, :d].astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def quantizer_topk(q, corpus_enc, queries_enc, k):
    cc = q.encode(corpus_enc)
    qc = q.encode(queries_enc)
    if hasattr(q, "hamming_distances"):
        dists = q.hamming_distances(qc, cc)
    else:
        # SignBitQuantizer exposes module-level helper
        import remax
        dists = remax.hamming_distances(qc, cc)
    # smallest distance = closest
    idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    row_d = np.take_along_axis(dists, idx, axis=1)
    order = np.argsort(row_d, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate(emb: np.ndarray, full_emb: np.ndarray):
    """Run a full {dim × ladder × R@K} sweep for one adapter's embeddings.

    Returns list of result dicts. `emb` is the (sliced+renormalized) embedding
    for the current dim. `full_emb` is the 768-d version (also normalized) for
    cross-dim ground truth.
    """
    rows = []
    n, d = emb.shape
    corpus, queries = split(emb, N_QUERIES, QUERY_SPLIT_SEED)
    full_corpus, full_queries = split(full_emb, N_QUERIES, QUERY_SPLIT_SEED)
    k_max = max(K_EVALS)
    truth_d = topk(corpus, queries, k_max)
    truth_full = topk(full_corpus, full_queries, k_max)

    # Pre-compute centered encoder inputs
    mu = corpus.mean(axis=0)
    corpus_c = corpus - mu
    queries_c = queries - mu

    # ---- float32 reference at this dim ----
    # float32 search at dim d, recall vs truth at dim d → 1.0 by definition
    # but recall vs truth at 768 measures matryoshka degradation.
    for K in K_EVALS:
        rows.append({
            "dim": d, "method": "float32", "ladder": "—",
            "vs": f"R@{K}@768",
            "recall": recall(truth_d, truth_full, K),
        })

    # ---- 1-bit ----
    q = SignBitQuantizer(d=d, seed=QUANTIZER_SEED)
    pred = quantizer_topk(q, corpus_c, queries_c, k_max)
    for K in K_EVALS:
        rows.append({
            "dim": d, "method": "1-bit", "ladder": "k=1",
            "vs": f"R@{K}@d",
            "recall": recall(pred, truth_d, K),
        })
        rows.append({
            "dim": d, "method": "1-bit", "ladder": "k=1",
            "vs": f"R@{K}@768",
            "recall": recall(pred, truth_full, K),
        })

    # ---- stacked ladder ----
    for k in LADDER_KS:
        q = StackedSignBitQuantizer(d=d, k=k, seed=QUANTIZER_SEED)
        pred = quantizer_topk(q, corpus_c, queries_c, k_max)
        for K in K_EVALS:
            rows.append({
                "dim": d, "method": "stacked", "ladder": f"k={k}",
                "vs": f"R@{K}@d",
                "recall": recall(pred, truth_d, K),
            })
            rows.append({
                "dim": d, "method": "stacked", "ladder": f"k={k}",
                "vs": f"R@{K}@768",
                "recall": recall(pred, truth_full, K),
            })

    return rows


def format_table(rows_for_task, task, K, ref_basis):
    """Build a markdown table for one adapter at a given K and reference basis.

    Columns: dim, float32 (matryoshka-only), 1-bit, k=2, k=4, k=8.
    `ref_basis` is "d" (quantizer-only) or "768" (end-to-end).
    """
    by_dim = {}
    for r in rows_for_task:
        if not r["vs"].endswith(f"@{ref_basis}"):
            continue
        if not r["vs"].startswith(f"R@{K}@"):
            continue
        by_dim.setdefault(r["dim"], {})[r["ladder"] if r["method"] != "float32" else "float32"] = r["recall"]

    lines = []
    lines.append(f"#### `{task}` — R@{K} vs float32@{ref_basis}")
    lines.append("")
    lines.append("| dim | float32 | 1-bit | k=2 | k=4 | k=8 |")
    lines.append("|----:|--------:|------:|----:|----:|----:|")
    for d in DIMS:
        c = by_dim.get(d, {})
        def fmt(v):
            return f"{v:.3f}" if v is not None else "—"
        # float32 only meaningful vs the @768 basis; vs @d it's 1.0 trivially
        f32 = c.get("float32")
        if ref_basis == "d":
            f32_str = "1.000"
        else:
            f32_str = fmt(f32)
        lines.append(
            f"| {d} | {f32_str} | {fmt(c.get('k=1'))} | {fmt(c.get('k=2'))} | "
            f"{fmt(c.get('k=4'))} | {fmt(c.get('k=8'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--out-md", type=str, default=str(RESULTS / "JINA_V5_NANO.md"))
    p.add_argument("--out-csv", type=str, default=str(RESULTS / "jina_v5_nano.csv"))
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = json.loads((CACHE / "meta.json").read_text())

    all_rows = {}
    for task in args.tasks:
        path = CACHE / f"{task}.npy"
        if not path.exists():
            sys.stderr.write(f"[skip] {task}: {path} not found\n")
            continue
        full_emb = np.load(path).astype(np.float32)
        sys.stderr.write(f"\n=== task={task}  shape={full_emb.shape} ===\n")
        # We need self-knn split at each dim; ground truth at @768 also uses
        # the same split — so we pre-renormalize once at 768 (no-op, already
        # unit) and pass that as full_emb to evaluate.
        task_rows = []
        for d in DIMS:
            t0 = time.time()
            emb_d = slice_renorm(full_emb, d)
            rows = evaluate(emb_d, full_emb)
            for r in rows:
                r["task"] = task
            task_rows.extend(rows)
            sys.stderr.write(f"  dim={d:3d}  {time.time()-t0:5.1f}s\n")
        all_rows[task] = task_rows

    # CSV
    flat = [r for rs in all_rows.values() for r in rs]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "dim", "method", "ladder", "vs", "recall"])
        w.writeheader()
        for r in flat:
            w.writerow({k: r[k] for k in w.fieldnames})

    # Markdown
    md = []
    md.append("# remax × jina-embeddings-v5-text-nano — matryoshka × adapter sweep")
    md.append("")
    md.append(f"- **Model**: `{meta['model']}` (EuroBERT-210M base, 238.9M params, 768 dim, last-token pool, L2-normalized)")
    md.append(f"- **Corpus**: SPECTER2 titles+abstracts (NLP-broad cache), n={meta['n']}, max_length={meta['max_length']} tokens, prompt_name=`document`")
    md.append(f"- **Held-out queries**: {N_QUERIES} via permutation seed {QUERY_SPLIT_SEED}")
    md.append(f"- **Matryoshka dims**: {list(DIMS)} (slice + L2-renorm — equivalent to passing `truncate_dim` to `model.encode`)")
    md.append(f"- **Quantizer**: centered SimHash (`corpus.mean` subtracted before encoding), seed {QUANTIZER_SEED}; native POPCNT scan")
    md.append("- **Two recall bases per cell**:")
    md.append("  * `R@K@d` — vs float32 ground truth at the **same** dim. Isolates quantizer quality.")
    md.append("  * `R@K@768` — vs float32 ground truth at the **full** 768 dim. Conflates matryoshka loss + quantizer loss (the user-facing number).")
    md.append("")
    for task in args.tasks:
        if task not in all_rows:
            continue
        md.append(f"## Task adapter: `{task}`")
        md.append("")
        for K in K_EVALS:
            md.append(format_table(all_rows[task], task, K, "d"))
        for K in K_EVALS:
            md.append(format_table(all_rows[task], task, K, "768"))

    Path(args.out_md).write_text("\n".join(md) + "\n")
    sys.stderr.write(f"\nwrote {args.out_md}\nwrote {args.out_csv}\n")


if __name__ == "__main__":
    main()
