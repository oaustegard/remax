"""NVFP4 vs BF16 embedding quality, and the remax composition test.

Two questions:
  1. What does NVIDIA's NVFP4 *model* quantization cost, measured on the emitted
     embeddings? (BF16-full vs NVFP4-full, absolute nDCG@10 / Spearman, plus how
     much the retrieval order shifts vs the BF16 float reference.)
  2. Does remax 1-bit *embedding* quantization compose with NVFP4? i.e. is
     remax-1-bit-on-NVFP4 ≈ remax-1-bit-on-BF16? The two quantizations act on
     different axes (model weights vs stored vectors); this checks they stack.

Everything is scored against ONE common reference — the BF16 full-float32 top-10
— so BF16 and NVFP4 rows are directly comparable.

Inputs: BF16 caches in $SCRATCH/emb, NVFP4 caches in $SCRATCH/emb_nvfp4.
Output: bench/results/nemotron_nvfp4.csv

Usage
-----
    python bench/nvfp4_eval.py
    python bench/nvfp4_eval.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_nemotron_1bit import (  # noqa: E402
    slice_renorm, r10_vs_float, ndcg_at_k, SCRATCH, DATA_DIR, RESULTS_DIR,
)
from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances  # noqa: E402

BF16_EMB = Path(os.environ.get("NEMOTRON_EMB_DIR", SCRATCH / "emb"))
NVFP4_EMB = Path(os.environ.get("NVFP4_EMB_DIR", SCRATCH / "emb_nvfp4"))
SEED = 0


def _bit_scores(quant, corpus, queries):
    codes = quant.encode(corpus)
    qcodes = quant.encode(queries)
    scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
    for i in range(queries.shape[0]):
        scores[i] = -hamming_distances(codes, qcodes[i])
    return scores, codes.shape[1]


def _methods(corpus, queries):
    """Yield (method, bytes_per_vec, score_matrix) for each embedding compressor."""
    yield "full_f32", corpus.shape[1] * 4, queries @ corpus.T
    s, b = _bit_scores(SignBitQuantizer(d=corpus.shape[1], seed=SEED), corpus, queries)
    yield "remax_1bit", b, s
    s, b = _bit_scores(StackedSignBitQuantizer(d=corpus.shape[1], k=4, seed=SEED), corpus, queries)
    yield "remax_stack4", b, s


def _pair_sim(method, s1, s2):
    if method == "full_f32":
        return np.sum(s1 * s2, axis=1), s1.shape[1] * 4
    d = s1.shape[1]
    quant = SignBitQuantizer(d=d, seed=SEED) if method == "remax_1bit" else StackedSignBitQuantizer(d=d, k=4, seed=SEED)
    c1, c2 = quant.encode(s1), quant.encode(s2)
    sim = np.array([-hamming_distances(c2, c1[i])[i] for i in range(len(c1))])
    return sim, c1.shape[1]


def run(bf16_dir: Path, nvfp4_dir: Path, out_csv: Path) -> None:
    scifact = json.load(open(DATA_DIR / "scifact_subset.json"))
    stsb = json.load(open(DATA_DIR / "stsb_test.json"))
    corpus_ids, query_ids = scifact["doc_ids"], scifact["query_ids"]
    qrels = scifact["qrels"]
    gold = np.array(stsb["score"], dtype=np.float32)

    emb = {}
    for tag, d in [("bf16", bf16_dir), ("nvfp4", nvfp4_dir)]:
        emb[tag] = {
            "docs": np.load(d / "scifact_docs.npy").astype(np.float32),
            "queries": np.load(d / "scifact_queries.npy").astype(np.float32),
            "s1": np.load(d / "stsb_s1.npy").astype(np.float32),
            "s2": np.load(d / "stsb_s2.npy").astype(np.float32),
        }

    # ONE common reference: BF16 full-float32 top-10.
    bf16_float_scores = emb["bf16"]["queries"] @ emb["bf16"]["docs"].T

    # headline agreement numbers
    doc_cos = float(np.mean(np.sum(emb["bf16"]["docs"] * emb["nvfp4"]["docs"], axis=1)))
    print(f"NVFP4 vs BF16 embedding agreement (SciFact docs): mean cosine = {doc_cos:.4f}")

    rows = []
    for tag in ("bf16", "nvfp4"):
        d = emb[tag]
        for method, bpv, scores in _methods(d["docs"], d["queries"]):
            r10 = r10_vs_float(scores, query_ids, corpus_ids, bf16_float_scores)
            ndcg = float(np.mean([
                ndcg_at_k(scores[i], corpus_ids, qrels.get(q, {}), 10)
                for i, q in enumerate(query_ids) if qrels.get(q)
            ]))
            rows.append(dict(dataset="scifact", embset=tag, method=method,
                             bytes_per_vec=bpv, r10_vs_bf16=round(r10, 4),
                             ndcg10=round(ndcg, 4), spearman=""))
        for method in ("full_f32", "remax_1bit", "remax_stack4"):
            sim, bpv = _pair_sim(method, d["s1"], d["s2"])
            rho = float(spearmanr(sim, gold).statistic)
            rows.append(dict(dataset="stsb", embset=tag, method=method,
                             bytes_per_vec=bpv, r10_vs_bf16="",
                             ndcg10="", spearman=round(rho, 4)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "embset", "method", "bytes_per_vec",
                                          "r10_vs_bf16", "ndcg10", "spearman"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    # readable summary
    print("\nSciFact (nDCG@10 vs qrels; R@10 vs BF16-float ref):")
    for r in rows:
        if r["dataset"] == "scifact":
            print(f"  {r['embset']:5s} {r['method']:13s} {r['bytes_per_vec']:5} B  "
                  f"nDCG={r['ndcg10']}  R@10vsBF16={r['r10_vs_bf16']}")
    print("STS-B (Spearman vs gold):")
    for r in rows:
        if r["dataset"] == "stsb":
            print(f"  {r['embset']:5s} {r['method']:13s} {r['bytes_per_vec']:5} B  rho={r['spearman']}")


def selftest() -> None:
    print("[nvfp4_eval] selftest ...")
    rng = np.random.default_rng(0)
    d = 256
    docs = rng.standard_normal((120, d)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = docs[:10] + 0.01 * rng.standard_normal((10, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ref = q @ docs.T
    ok = 0
    for method, bpv, scores in _methods(docs, q):
        r10 = r10_vs_float(scores, [str(i) for i in range(10)], [str(i) for i in range(120)], ref)
        assert 0.0 <= r10 <= 1.0, (method, r10)
        if method == "full_f32":
            assert r10 == 1.0
        ok += 1
    print(f"PASS: {ok} methods, full_f32 r10==1.0, all in [0,1]")
    sim, bpv = _pair_sim("remax_1bit", docs[:50], docs[:50])
    assert np.all(sim == 0), "identical pairs must have 0 Hamming distance"
    print("PASS: identical-pair Hamming distance is 0")
    print("[nvfp4_eval] selftest: passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    run(BF16_EMB, NVFP4_EMB, RESULTS_DIR / "nemotron_nvfp4.csv")


if __name__ == "__main__":
    main()
