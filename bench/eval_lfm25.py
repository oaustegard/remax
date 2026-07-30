"""Codec bake-off for LFM2.5-Embedding-350M on BEIR SciFact.

Mirrors the NEMOTRON_MASTER methodology: every codec is scored on real qrels
(nDCG@10 / R@10 / R@100) *and* on agreement with the fp32 ranking, then laid out
against bytes-per-vector so the comparison is at matched storage cost rather
than matched bit-width.

Inputs are plain .npy — this script never touches a model. Produce the vectors
with bench/embed_lfm25.py first.

    python bench/eval_lfm25.py --emb-dir bench/.cache/LFM25_SCIFACT \
                               --data-dir <scifact json dir> \
                               --out bench/results

Codec families compared:
  fp32            full and prefix-truncated (the Matryoshka-style control)
  remax           centered SimHash, 1-bit and stacked k=2/4/8
  remax-uncentered  1-bit without mean subtraction (isolates centering's value)
  remex           Lloyd-Max scalar quant at 1/2/3/4/8 bits
  int8            per-vector symmetric scalar quant (the conventional baseline)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from remax import SignBitQuantizer, StackedSignBitQuantizer  # noqa: E402
from remax.packing import hamming_distances  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geometry import describe, format_report, predict  # noqa: E402

K_NDCG = 10
DIMS_DEFAULT = (1024, 512, 256, 128, 64)
REMEX_BITS = (1, 2, 3, 4, 8)
STACK_KS = (2, 4, 8)


# ---------------------------------------------------------------- metrics


def ndcg_at_k(score_row, corpus_ids, rel_q, k=K_NDCG) -> float:
    top = np.argsort(-score_row, kind="stable")[:k]
    gains = [rel_q.get(corpus_ids[i], 0) for i in top]
    dcg = sum((2**g - 1) / np.log2(r + 2) for r, g in enumerate(gains))
    ideal = sorted(rel_q.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(score_row, corpus_ids, rel_q, k) -> float:
    top = np.argsort(-score_row, kind="stable")[:k]
    hits = sum(1 for i in top if rel_q.get(corpus_ids[i], 0) > 0)
    n_rel = sum(1 for v in rel_q.values() if v > 0)
    return hits / n_rel if n_rel > 0 else 0.0


def rank_agreement(scores, fp32_scores, k=10) -> float:
    """Fraction of the fp32 top-k that the codec also puts in its top-k.

    Complements the qrels metrics: nDCG can survive a codec that reshuffles the
    ranking as long as it keeps judged docs near the top, so this measures
    fidelity to the uncompressed ranking directly.
    """
    out = []
    for i in range(scores.shape[0]):
        a = set(np.argsort(-scores[i], kind="stable")[:k].tolist())
        b = set(np.argsort(-fp32_scores[i], kind="stable")[:k].tolist())
        out.append(len(a & b) / k)
    return float(np.mean(out))


def evaluate(scores, corpus_ids, query_ids, rel, fp32_scores=None) -> dict:
    nd, r10, r100 = [], [], []
    for qi, qid in enumerate(query_ids):
        rq = rel.get(qid, {})
        if not rq:
            continue
        nd.append(ndcg_at_k(scores[qi], corpus_ids, rq, K_NDCG))
        r10.append(recall_at_k(scores[qi], corpus_ids, rq, 10))
        r100.append(recall_at_k(scores[qi], corpus_ids, rq, 100))
    out = {
        "ndcg@10": float(np.mean(nd)),
        "r@10": float(np.mean(r10)),
        "r@100": float(np.mean(r100)),
    }
    if fp32_scores is not None:
        out["agree@10"] = rank_agreement(scores, fp32_scores, 10)
    return out


# ---------------------------------------------------------------- codecs


def _unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def slice_renorm(X, d):
    """Matryoshka-style prefix truncation, renormalized."""
    return _unit(X[:, :d].astype(np.float32))


def fp32_scores(C, Q):
    return (Q @ C.T).astype(np.float64)


def remax_scores(C, Q, *, k=None, seed=42, center=True):
    """Centered SimHash. k=None is the plain 1-bit quantizer."""
    mu = C.mean(axis=0) if center else np.zeros(C.shape[1], dtype=C.dtype)
    Cc, Qc = (C - mu).astype(np.float32), (Q - mu).astype(np.float32)
    d = C.shape[1]
    quant = (
        SignBitQuantizer(d=d, seed=seed)
        if k is None
        else StackedSignBitQuantizer(d=d, k=k, seed=seed)
    )
    c_codes, q_codes = quant.encode(Cc), quant.encode(Qc)
    out = np.empty((len(q_codes), len(c_codes)), dtype=np.float64)
    for i in range(len(q_codes)):
        out[i] = -hamming_distances(c_codes, q_codes[i])
    return out, c_codes.shape[1]


def remex_scores(C, Q, *, bits, seed=42):
    import remex

    d = C.shape[1]
    quant = remex.Quantizer(d=d, bits=bits, seed=seed)
    cv = quant.encode(C.astype(np.float32))
    chat = quant.decode(cv)
    scores = (Q.astype(np.float32) @ chat.T).astype(np.float64)
    # bytes/row excluding the float32 norm remex stores alongside each code
    return scores, (d * bits + 7) // 8


def int8_scores(C, Q):
    """Per-vector symmetric int8 — the conventional 4x baseline."""
    scale = np.abs(C).max(axis=1, keepdims=True) / 127.0
    q = np.round(C / np.maximum(scale, 1e-12)).clip(-127, 127).astype(np.int8)
    chat = q.astype(np.float32) * scale
    return (Q.astype(np.float32) @ chat.T).astype(np.float64), C.shape[1]


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="*", default=list(DIMS_DEFAULT))
    ap.add_argument("--label", default="LFM2.5-Embedding-350M")
    args = ap.parse_args()

    emb, data = pathlib.Path(args.emb_dir), pathlib.Path(args.data_dir)
    C_raw = np.load(emb / "docs.npy")
    Q_raw = np.load(emb / "queries.npy")
    corpus_ids = json.loads((emb / "doc_ids.json").read_text())
    query_ids = json.loads((emb / "query_ids.json").read_text())
    rel = json.loads((data / "qrels.json").read_text())

    print(f"corpus {C_raw.shape}  queries {Q_raw.shape}  qrels {len(rel)}")
    full_d = C_raw.shape[1]

    # --- geometry first, so the prediction is on the record before results ---
    g = describe(C_raw, seed=args.seed)
    p = predict(g)
    geom_md = format_report(args.label, g, p)
    print("\n" + geom_md + "\n")

    C, Q = _unit(C_raw.astype(np.float32)), _unit(Q_raw.astype(np.float32))
    base = fp32_scores(C, Q)
    rows = []

    def add(label, family, dim, scores, bytes_row, t0):
        m = evaluate(scores, corpus_ids, query_ids, rel, fp32_scores=base)
        m.update(
            label=label, family=family, dim=dim,
            bytes_per_vec=int(bytes_row), secs=round(time.time() - t0, 2),
        )
        rows.append(m)
        print(
            f"  {label:<28} {bytes_row:>6}B  nDCG@10 {m['ndcg@10']:.4f}  "
            f"R@10 {m['r@10']:.4f}  R@100 {m['r@100']:.4f}  agree@10 {m['agree@10']:.4f}"
        )

    print("\n--- fp32 (full + prefix truncation) ---")
    for d in args.dims:
        if d > full_d:
            continue
        t0 = time.time()
        s = base if d == full_d else fp32_scores(slice_renorm(C, d), slice_renorm(Q, d))
        add(f"fp32 d={d}", "fp32", d, s, d * 4, t0)

    print("\n--- remax (centered SimHash) ---")
    for d in args.dims:
        if d > full_d or d % 8:
            continue
        t0 = time.time()
        s, b = remax_scores(slice_renorm(C, d), slice_renorm(Q, d), seed=args.seed)
        add(f"remax 1bit d={d}", "remax", d, s, b, t0)
    for k in STACK_KS:
        t0 = time.time()
        s, b = remax_scores(C, Q, k=k, seed=args.seed)
        add(f"remax k={k} d={full_d}", "remax", full_d, s, b, t0)

    print("\n--- remax uncentered (ablation) ---")
    t0 = time.time()
    s, b = remax_scores(C, Q, seed=args.seed, center=False)
    add(f"remax 1bit uncentered d={full_d}", "remax-unc", full_d, s, b, t0)

    print("\n--- remex (Lloyd-Max scalar quant) ---")
    for bits in REMEX_BITS:
        t0 = time.time()
        try:
            s, b = remex_scores(C, Q, bits=bits, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  remex {bits}bit FAILED: {exc}")
            continue
        add(f"remex {bits}bit d={full_d}", "remex", full_d, s, b, t0)

    print("\n--- int8 baseline ---")
    t0 = time.time()
    s, b = int8_scores(C, Q)
    add(f"int8 d={full_d}", "int8", full_d, s, b, t0)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lfm25_scifact.json").write_text(
        json.dumps({"geometry": g, "prediction": p, "rows": rows}, indent=2)
    )

    fp32_full = next(r for r in rows if r["label"] == f"fp32 d={full_d}")
    lines = [
        f"# LFM2.5 codec bake-off — BEIR SciFact",
        "",
        f"Corpus {C_raw.shape[0]} docs / {Q_raw.shape[0]} queries / "
        f"{sum(len(v) for v in rel.values())} judgments. Seed {args.seed}.",
        "",
        geom_md,
        "",
        "## Results (sorted by bytes/vector)",
        "",
        "| codec | B/vec | ratio | nDCG@10 | dnDCG | R@10 | R@100 | agree@10 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(rows, key=lambda r: (r["bytes_per_vec"], -r["ndcg@10"])):
        lines.append(
            f"| {r['label']} | {r['bytes_per_vec']} | "
            f"{full_d * 4 / r['bytes_per_vec']:.1f}x | {r['ndcg@10']:.4f} | "
            f"{r['ndcg@10'] - fp32_full['ndcg@10']:+.4f} | {r['r@10']:.4f} | "
            f"{r['r@100']:.4f} | {r['agree@10']:.4f} |"
        )
    (outdir / "LFM25_SCIFACT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outdir/'LFM25_SCIFACT.md'} and {outdir/'lfm25_scifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
