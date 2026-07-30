"""Two-stage retrieval for LFM2.5 on SciFact: cheap codec shortlist, fp32 rescore.

A single-stage quantization table understates what quantization is actually worth,
because nobody deploys the coarse codes alone. The deployable pattern is: scan the
compressed index for a shortlist of N, then rescore that shortlist against exact
vectors. The index stays small; the fp32 cost is paid on N rows, not the corpus.

bench/results/RERANK.md already establishes this on SPECTER2 (1-bit R@10 0.635 ->
0.983 after fp32 rescore of the top-100). This script asks the same question of
LFM2.5, and reports the shortlist depth at which quantization stops costing
anything measurable -- which is the number you actually need to size an index.

    python bench/rerank_lfm25.py --emb-dir bench/.cache/LFM25_SCIFACT \
                                 --data-dir <scifact json dir> --out bench/results
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_lfm25 import (  # noqa: E402
    _unit, fp32_scores, int8_scores, ndcg_at_k, recall_at_k,
    remax_scores, remex_scores,
)

DEPTHS = (10, 25, 50, 100, 200, 500, 1000)


def stage1_variants(C, Q, seed):
    """(label, bytes_per_vec, score_matrix) for each shortlist codec worth testing."""
    out = []
    s, b = remax_scores(C, Q, seed=seed)
    out.append(("remax 1bit", b, s))
    for k in (2, 4):
        s, b = remax_scores(C, Q, k=k, seed=seed)
        out.append((f"remax k={k}", b, s))
    for bits in (1, 2, 4):
        try:
            s, b = remex_scores(C, Q, bits=bits, seed=seed)
            out.append((f"remex {bits}bit", b, s))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip remex {bits}bit: {exc}")
    s, b = int8_scores(C, Q)
    out.append(("int8", b, s))
    return out


def rescore(coarse, exact, depth):
    """Take each query's top-`depth` by coarse score, reorder those by exact score.

    Everything outside the shortlist keeps a score below the rescored block, so the
    result is a full ranking whose head is exact and whose tail is coarse -- exactly
    what a real two-stage index returns.
    """
    n_q, n_d = coarse.shape
    out = np.full((n_q, n_d), -np.inf, dtype=np.float64)
    d = min(depth, n_d)
    for i in range(n_q):
        cand = np.argpartition(-coarse[i], d - 1)[:d]
        out[i, cand] = exact[i, cand]
    return out


def score(scores, corpus_ids, query_ids, rel):
    nd, r10 = [], []
    for qi, qid in enumerate(query_ids):
        rq = rel.get(qid, {})
        if not rq:
            continue
        nd.append(ndcg_at_k(scores[qi], corpus_ids, rq, 10))
        r10.append(recall_at_k(scores[qi], corpus_ids, rq, 10))
    return float(np.mean(nd)), float(np.mean(r10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depths", type=int, nargs="*", default=list(DEPTHS))
    args = ap.parse_args()

    emb, data = pathlib.Path(args.emb_dir), pathlib.Path(args.data_dir)
    C = _unit(np.load(emb / "docs.npy").astype(np.float32))
    Q = _unit(np.load(emb / "queries.npy").astype(np.float32))
    corpus_ids = json.loads((emb / "doc_ids.json").read_text())
    query_ids = json.loads((emb / "query_ids.json").read_text())
    rel = json.loads((data / "qrels.json").read_text())

    exact = fp32_scores(C, Q)
    base_nd, base_r10 = score(exact, corpus_ids, query_ids, rel)
    full_bytes = C.shape[1] * 4
    print(f"fp32 baseline: nDCG@10 {base_nd:.4f}  R@10 {base_r10:.4f}  "
          f"({full_bytes} B/vec)\n")

    rows = []
    for label, bpv, coarse in stage1_variants(C, Q, args.seed):
        t0 = time.time()
        nd0, r0 = score(coarse, corpus_ids, query_ids, rel)
        line = {"codec": label, "bytes_per_vec": int(bpv),
                "ndcg@10_stage1": nd0, "r@10_stage1": r0, "depths": {}}
        print(f"{label:<14} {bpv:>5}B  stage1 nDCG@10 {nd0:.4f} "
              f"({nd0 - base_nd:+.4f})")
        for d in args.depths:
            nd, r = score(rescore(coarse, exact, d),
                          corpus_ids, query_ids, rel)
            line["depths"][str(d)] = {"ndcg@10": nd, "r@10": r}
            frac = nd / base_nd if base_nd else 0.0
            print(f"    +fp32 rescore top-{d:<5} nDCG@10 {nd:.4f} "
                  f"({nd - base_nd:+.4f}, {frac*100:.1f}% of fp32)")
        line["secs"] = round(time.time() - t0, 1)
        rows.append(line)
        print()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"fp32": {"ndcg@10": base_nd, "r@10": base_r10,
                        "bytes_per_vec": full_bytes},
               "depths": args.depths, "rows": rows}
    (outdir / "lfm25_rerank.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# LFM2.5 two-stage retrieval — BEIR SciFact",
        "",
        f"fp32 baseline nDCG@10 **{base_nd:.4f}** at {full_bytes} B/vec.",
        "",
        "Shortlist comes from the compressed index; the shortlist is then reordered",
        "by exact fp32 score. Percentages are of the fp32 baseline.",
        "",
        "| codec | B/vec | stage-1 | " + " | ".join(f"top-{d}" for d in args.depths) + " |",
        "|---|--:|--:|" + "--:|" * len(args.depths),
    ]
    for r in sorted(rows, key=lambda r: r["bytes_per_vec"]):
        cells = " | ".join(
            f"{r['depths'][str(d)]['ndcg@10']:.4f}" for d in args.depths
        )
        md.append(
            f"| {r['codec']} | {r['bytes_per_vec']} | "
            f"{r['ndcg@10_stage1']:.4f} | {cells} |"
        )
    (outdir / "LFM25_RERANK.md").write_text("\n".join(md) + "\n")
    print(f"wrote {outdir/'LFM25_RERANK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
