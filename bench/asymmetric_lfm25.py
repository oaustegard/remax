"""Asymmetric vs symmetric 1-bit scoring — testing Exa's design choice on our data.

Exa's web-scale index (exa.ai/blog/building-web-scale-vector-db) stores documents
as sign bits but keeps the QUERY in float, scoring float-query against the +/-1
values the bits stand for rather than comparing two bit strings. remax's
``search`` binarizes both sides.

That asymmetry is free. The query is one vector per search -- it occupies no
index storage -- so if it buys accuracy, symmetric Hamming is discarding it for
nothing.

Our SciFact run hinted at this without naming it: at an identical 128 B/vec,
remex 1-bit scored 0.7018 and remax 1-bit 0.6772. remex decodes to float and
takes an inner product; remax compares bit strings. This script isolates the
single variable -- same centering, same rotation, same document bits, only the
query side differs -- so the gap is attributable to query precision alone.

Everything routes through the library API (SignBitQuantizer.encode and
remax.asymmetric_scores / hamming_distances) rather than a hand-rolled rotation,
so the symmetric column here reproduces eval_lfm25's remax rows exactly.

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

from remax import (  # noqa: E402
    SignBitQuantizer, StackedSignBitQuantizer, asymmetric_scores,
)
from remax.packing import hamming_distances  # noqa: E402
from eval_lfm25 import _unit, evaluate, fp32_scores, slice_renorm  # noqa: E402

DIMS = (1024, 512, 256, 128)
STACK_KS = (1, 2, 4)


def score_matrices(C, Q, d, seed):
    """(symmetric, asymmetric) score matrices over the SAME encoded corpus.

    Centering mirrors eval_lfm25.remax_scores: subtract the corpus mean from
    both sides before encoding. The document codes are produced once and shared,
    so the two matrices differ only in how the query is treated.
    """
    Cd, Qd = slice_renorm(C, d), slice_renorm(Q, d)
    mu = Cd.mean(axis=0)
    Cc, Qc = (Cd - mu).astype(np.float32), (Qd - mu).astype(np.float32)

    quant = SignBitQuantizer(d=d, seed=seed)
    c_codes = quant.encode(Cc)          # (n, d/8) -- the stored index
    q_codes = quant.encode(Qc)          # symmetric path binarizes the query
    q_rot = Qc @ quant.rotation_        # asymmetric path keeps it in float

    sym = np.empty((len(Qc), len(Cc)), dtype=np.float64)
    asym = np.empty_like(sym)
    for i in range(len(Qc)):
        sym[i] = -hamming_distances(c_codes, q_codes[i])
        asym[i] = asymmetric_scores(q_rot[i], c_codes)
    return sym, asym, c_codes.shape[1]


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
        sym, asym, bpv = score_matrices(C, Q, d, args.seed)
        ms = evaluate(sym, corpus_ids, query_ids, rel, fp32_scores=base)
        ma = evaluate(asym, corpus_ids, query_ids, rel, fp32_scores=base)
        print(f"{d:>5} {bpv:>6} {ms['ndcg@10']:>10.4f} {ma['ndcg@10']:>11.4f} "
              f"{ma['ndcg@10'] - ms['ndcg@10']:>+8.4f} "
              f"{ms['agree@10']:>10.4f} {ma['agree@10']:>11.4f}")
        rows.append({"dim": d, "bytes_per_vec": int(bpv),
                     "symmetric": ms, "asymmetric": ma})

    # Stacking and asymmetry are two ways to buy down the same error: the
    # variance of a similarity estimated from sign bits. Stacking pays index
    # bytes for it, asymmetry pays nothing. If they are substitutes rather than
    # complements, the gain should collapse once the stack is deep -- and the
    # cheap fix should be competitive with the expensive one.
    print(f"\n{'codec':<14} {'B/vec':>6} {'symmetric':>10} {'asymmetric':>11} "
          f"{'delta':>8}")
    print("-" * 54)
    mu = C.mean(axis=0)
    Cc, Qc = (C - mu).astype(np.float32), (Q - mu).astype(np.float32)
    stacked = []
    for k in STACK_KS:
        if k == 1:
            quant, rot = SignBitQuantizer(d=C.shape[1], seed=args.seed), "rotation_"
        else:
            quant = StackedSignBitQuantizer(d=C.shape[1], k=k, seed=args.seed)
            rot = "_rotation_matrix"
        c_codes, q_codes = quant.encode(Cc), quant.encode(Qc)
        q_rot = Qc @ getattr(quant, rot)
        sym = np.empty((len(Qc), len(Cc)), dtype=np.float64)
        asym = np.empty_like(sym)
        for i in range(len(Qc)):
            sym[i] = -hamming_distances(c_codes, q_codes[i])
            asym[i] = asymmetric_scores(q_rot[i], c_codes)
        ms = evaluate(sym, corpus_ids, query_ids, rel, fp32_scores=base)
        ma = evaluate(asym, corpus_ids, query_ids, rel, fp32_scores=base)
        bpv = c_codes.shape[1]
        print(f"{'k=' + str(k):<14} {bpv:>6} {ms['ndcg@10']:>10.4f} "
              f"{ma['ndcg@10']:>11.4f} {ma['ndcg@10'] - ms['ndcg@10']:>+8.4f}")
        stacked.append({"k": k, "bytes_per_vec": int(bpv),
                        "symmetric": ms, "asymmetric": ma})

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lfm25_asymmetric.json").write_text(
        json.dumps({"fp32": fp32, "rows": rows, "stacked": stacked}, indent=2))
    print(f"\nwrote {outdir / 'lfm25_asymmetric.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
