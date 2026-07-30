"""Asymmetric vs symmetric 1-bit scoring — testing Exa's design choice on our data.

Exa's web-scale index (exa.ai/blog/building-web-scale-vector-db) stores documents
as sign bits but keeps the QUERY in float, scoring float-query against +/-1 doc by
dot product rather than Hamming. remax binarizes both sides.

That asymmetry is free. The query is one vector per search -- it costs no index
storage to keep it exact -- so if it buys accuracy, symmetric Hamming is leaving
it on the floor for nothing.

Our SciFact run already hints at this without naming it: at an identical 128 B/vec,
remex 1-bit scored 0.7018 and remax 1-bit 0.6772. remex decodes to float and takes
an inner product; remax compares two bit strings. This script isolates that single
variable -- same centering, same Haar rotation, same sign bits on the document
side -- so the gap is attributable to query precision alone and nothing else.

    python bench/asymmetric_lfm25.py --emb-dir bench/.cache/LFM25_SCIFACT \
                                     --data-dir <scifact json dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from remax.rotation import haar_rotation  # noqa: E402
from eval_lfm25 import _unit, evaluate, fp32_scores, slice_renorm  # noqa: E402

DIMS = (1024, 512, 256, 128)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    emb, data = pathlib.Path(args.emb_dir), pathlib.Path(args.data_dir)
    C = _unit(np.load(emb / "docs.npy").astype(np.float32))
    Q = _unit(np.load(emb / "queries.npy").astype(np.float32))
    corpus_ids = json.loads((emb / "doc_ids.json").read_text())
    query_ids = json.loads((emb / "query_ids.json").read_text())
    rel = json.loads((data / "qrels.json").read_text())

    base = fp32_scores(C, Q)
    fp32 = evaluate(base, corpus_ids, query_ids, rel, fp32_scores=base)
    print(f"fp32 d={C.shape[1]}: nDCG@10 {fp32['ndcg@10']:.4f}\n")

    rows = []
    print(f"{'dim':>5} {'B/vec':>6} {'symmetric':>10} {'asymmetric':>11} "
          f"{'delta':>8} {'sym agree':>10} {'asym agree':>11}")
    print("-" * 70)
    for d in DIMS:
        Cd, Qd = slice_renorm(C, d), slice_renorm(Q, d)
        mu = Cd.mean(axis=0)
        Cc, Qc = Cd - mu, Qd - mu

        R = haar_rotation(d, seed=args.seed).astype(np.float32)
        Cr, Qr = Cc @ R.T, Qc @ R.T

        # Document side is identical in both: one sign bit per dimension.
        Cb = np.where(Cr > 0, 1.0, -1.0).astype(np.float32)

        # Symmetric: query binarized too. Equivalent to Hamming up to an affine
        # map (dot of two +/-1 vectors == d - 2*hamming), so ranking is identical.
        Qb = np.where(Qr > 0, 1.0, -1.0).astype(np.float32)
        sym = (Qb @ Cb.T).astype(np.float64)

        # Asymmetric: query stays float. Exa's choice.
        asym = (Qr @ Cb.T).astype(np.float64)

        ms = evaluate(sym, corpus_ids, query_ids, rel, fp32_scores=base)
        ma = evaluate(asym, corpus_ids, query_ids, rel, fp32_scores=base)
        bpv = d // 8
        print(f"{d:>5} {bpv:>6} {ms['ndcg@10']:>10.4f} {ma['ndcg@10']:>11.4f} "
              f"{ma['ndcg@10'] - ms['ndcg@10']:>+8.4f} "
              f"{ms['agree@10']:>10.4f} {ma['agree@10']:>11.4f}")
        rows.append({"dim": d, "bytes_per_vec": bpv,
                     "symmetric": ms, "asymmetric": ma})

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lfm25_asymmetric.json").write_text(
        json.dumps({"fp32": fp32, "rows": rows}, indent=2))
    print(f"\nwrote {outdir / 'lfm25_asymmetric.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
