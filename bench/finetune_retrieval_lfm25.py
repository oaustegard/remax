"""Is a retrieval fine-tune on SciFact worthwhile? Measured, with a real holdout.

The classification cookbook fine-tunes on thousands of labelled documents. A
retrieval fine-tune trains on (query, relevant-document) pairs instead, and
SciFact has 339 of them across 300 queries -- so the training set is roughly two
orders of magnitude smaller than the compute budget would suggest. That makes
overfitting, not compute, the thing to measure.

DESIGN, in the order the decisions matter:

  * The split is at QUERY level, seeded, and the validation queries are never
    seen in any form during training -- not as anchors, not as in-batch
    negatives. A document-level split would leak, because the same abstract can
    be relevant to a training query and a validation query.
  * Train AND validation metrics are both reported for both models. A fine-tune
    on 225 pairs will improve training-set retrieval essentially for free; the
    only interesting number is what happens to the held-out queries, and the gap
    between the two is the overfitting readout.
  * The evaluation corpus is fixed once and shared by both models: every judged
    document plus random distractors up to --corpus-size. Base-model vectors for
    it come from the existing cache, so only the fine-tuned model pays an encode.
  * Quantized retrieval is evaluated alongside fp32. If a fine-tune helps the
    1-bit index more than it helps fp32, it has learned something
    quantization-friendly; if less, it has sharpened distinctions that
    binarization then throws away.

Loss is in-batch-negatives cross-entropy (MultipleNegativesRanking): the other
documents in the batch are the negatives. With a batch of 8 that is 7 negatives
per anchor, which is weak -- real recipes use 64+. Stated because it caps what
this experiment can show, and no batch bigger than 8 fits in 15 GB here.

    python bench/finetune_retrieval_lfm25.py --data-dir <scifact json dir> \
        --emb-dir bench/.cache/LFM25_SCIFACT --epochs 3
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

from remax.rotation import haar_rotation  # noqa: E402
from remax.packing import encode_signs, hamming_distances, asymmetric_scores  # noqa: E402
from eval_lfm25 import _unit, evaluate, fp32_scores  # noqa: E402

MODEL = "LiquidAI/LFM2.5-Embedding-350M"
PROMPTS = {"query": "query: ", "document": "document: "}


# ---------------------------------------------------------------- data


def split_queries(qrels, queries, n_val, seed):
    """Query-level split. Validation queries never appear during training."""
    have_pos = [q["id"] for q in queries
                if any(v > 0 for v in qrels.get(q["id"], {}).values())]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(have_pos))
    val = {have_pos[i] for i in perm[:n_val]}
    train = {have_pos[i] for i in perm[n_val:]}
    return train, val


def build_corpus(qrels, docs, size, seed):
    """Judged documents plus random distractors, fixed for both models."""
    judged = {c for m in qrels.values() for c, v in m.items() if v > 0}
    by_id = {d["id"]: d for d in docs}
    keep = [i for i in judged if i in by_id]
    rest = [d["id"] for d in docs if d["id"] not in judged]
    rng = np.random.default_rng(seed)
    extra = rng.permutation(len(rest))[: max(0, size - len(keep))]
    keep += [rest[i] for i in extra]
    return [by_id[i] for i in keep]


# ---------------------------------------------------------------- model


def load(lora_r, seed):
    import torch
    from transformers import AutoModel

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  dtype=torch.float32)
    if lora_r:
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                            "in_proj", "w1", "w2", "w3"]))
    return m


def embed(model, tok, texts, prompt, max_len, batch=8, grad=False):
    import torch

    out = []
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        for i in range(0, len(texts), batch):
            enc = tok([PROMPTS[prompt] + t for t in texts[i:i + batch]],
                      padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            h = model(**enc).last_hidden_state[:, 0]      # CLS, as upstream
            out.append(torch.nn.functional.normalize(h, p=2, dim=1))
    return torch.cat(out) if len(out) > 1 else out[0]


def finetune(model, tok, pairs, *, epochs, lr, batch, qlen, dlen, temp, seed):
    """In-batch-negatives cross-entropy over (query, positive) pairs."""
    import torch

    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    hist = []
    for ep in range(epochs):
        order = torch.randperm(len(pairs), generator=g).tolist()
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(order) - 1, batch):
            chunk = [pairs[j] for j in order[i:i + batch]]
            if len(chunk) < 2:
                continue  # in-batch negatives need at least one other row
            Q = embed(model, tok, [c[0] for c in chunk], "query", qlen,
                      batch=len(chunk), grad=True)
            D = embed(model, tok, [c[1] for c in chunk], "document", dlen,
                      batch=len(chunk), grad=True)
            logits = (Q @ D.T) / temp
            loss = lossf(logits, torch.arange(len(chunk)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1),
                     "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    epoch {ep + 1}: loss {hist[-1]['loss']:.4f} "
              f"({hist[-1]['minutes']} min)")
    model.eval()
    return hist


# ---------------------------------------------------------------- eval


def all_metrics(C, Q, cids, qids, rel, seed):
    """fp32 plus 1-bit symmetric/asymmetric on the same vectors."""
    C, Q = _unit(C), _unit(Q)
    base = fp32_scores(C, Q)
    out = {"fp32": evaluate(base, cids, qids, rel, fp32_scores=base)}
    mu = C.mean(axis=0)
    Cc, Qc = (C - mu).astype(np.float32), (Q - mu).astype(np.float32)
    R = haar_rotation(C.shape[1], seed=seed).astype(np.float32)
    cc, qc, qr = encode_signs(Cc @ R), encode_signs(Qc @ R), Qc @ R
    sym = np.empty((len(Qc), len(Cc))); asym = np.empty_like(sym)
    for i in range(len(Qc)):
        sym[i] = -hamming_distances(cc, qc[i])
        asym[i] = asymmetric_scores(qr[i], cc)
    out["1bit_sym"] = evaluate(sym, cids, qids, rel, fp32_scores=base)
    out["1bit_asym"] = evaluate(asym, cids, qids, rel, fp32_scores=base)
    return out


def show(tag, m):
    print(f"  {tag:<26} fp32 {m['fp32']['ndcg@10']:.4f}   "
          f"1bit-sym {m['1bit_sym']['ndcg@10']:.4f}   "
          f"1bit-asym {m['1bit_asym']['ndcg@10']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16, help="0 = full fine-tune")
    ap.add_argument("--qlen", type=int, default=64)
    ap.add_argument("--dlen", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--corpus-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data, emb = pathlib.Path(args.data_dir), pathlib.Path(args.emb_dir)
    docs = json.loads((data / "docs.json").read_text())
    queries = json.loads((data / "queries.json").read_text())
    qrels = json.loads((data / "qrels.json").read_text())

    tr_q, va_q = split_queries(qrels, queries, args.n_val, args.seed)
    corpus = build_corpus(qrels, docs, args.corpus_size, args.seed)
    cids = [d["id"] for d in corpus]
    by_q = {q["id"]: q["text"] for q in queries}
    by_d = {d["id"]: d["text"] for d in docs}

    pairs = [(by_q[q], by_d[c]) for q in tr_q
             for c, v in qrels[q].items() if v > 0 and c in by_d]
    print(f"train {len(tr_q)} queries / {len(pairs)} pairs   "
          f"val {len(va_q)} queries (held out)   corpus {len(cids)} docs\n")

    tr_ids, va_ids = sorted(tr_q), sorted(va_q)
    tr_txt = [by_q[q] for q in tr_ids]
    va_txt = [by_q[q] for q in va_ids]

    # --- base model: reuse the cache, no encode needed -----------------
    cache_ids = json.loads((emb / "doc_ids.json").read_text())
    pos_of = {d: i for i, d in enumerate(cache_ids)}
    Cb = np.load(emb / "docs.npy")[[pos_of[c] for c in cids]]
    qcache = json.loads((emb / "query_ids.json").read_text())
    qpos = {q: i for i, q in enumerate(qcache)}
    Qall = np.load(emb / "queries.npy")
    base_tr = all_metrics(Cb, Qall[[qpos[q] for q in tr_ids]], cids, tr_ids,
                          qrels, args.seed)
    base_va = all_metrics(Cb, Qall[[qpos[q] for q in va_ids]], cids, va_ids,
                          qrels, args.seed)
    print("BASE (no fine-tune)")
    show("train queries", base_tr)
    show("val queries (held out)", base_va)

    # --- fine-tune ------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load(args.lora_r, args.seed)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFINE-TUNE ({'LoRA r=%d' % args.lora_r if args.lora_r else 'full'}, "
          f"{n_tr:,} trainable, {args.epochs} epochs, lr {args.lr})")
    hist = finetune(model, tok, pairs, epochs=args.epochs, lr=args.lr,
                    batch=args.batch, qlen=args.qlen, dlen=args.dlen,
                    temp=args.temp, seed=args.seed)

    print("\n  re-encoding evaluation corpus with the fine-tuned model...")
    t0 = time.time()
    Cf = embed(model, tok, [by_d[c] for c in cids], "document", args.dlen).numpy()
    Qtr = embed(model, tok, tr_txt, "query", args.qlen).numpy()
    Qva = embed(model, tok, va_txt, "query", args.qlen).numpy()
    print(f"  encoded in {(time.time() - t0) / 60:.1f} min")

    ft_tr = all_metrics(Cf, Qtr, cids, tr_ids, qrels, args.seed)
    ft_va = all_metrics(Cf, Qva, cids, va_ids, qrels, args.seed)
    print("\nFINE-TUNED")
    show("train queries", ft_tr)
    show("val queries (held out)", ft_va)

    print("\nDELTA (fine-tuned - base)")
    for tag, b, f in (("train", base_tr, ft_tr), ("val  ", base_va, ft_va)):
        print(f"  {tag}  fp32 {f['fp32']['ndcg@10'] - b['fp32']['ndcg@10']:+.4f}   "
              f"1bit-sym {f['1bit_sym']['ndcg@10'] - b['1bit_sym']['ndcg@10']:+.4f}   "
              f"1bit-asym {f['1bit_asym']['ndcg@10'] - b['1bit_asym']['ndcg@10']:+.4f}")
    gap_b = base_tr["fp32"]["ndcg@10"] - base_va["fp32"]["ndcg@10"]
    gap_f = ft_tr["fp32"]["ndcg@10"] - ft_va["fp32"]["ndcg@10"]
    print(f"\n  train-minus-val gap (fp32): base {gap_b:+.4f} -> "
          f"fine-tuned {gap_f:+.4f}   (widening = overfitting)")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lfm25_finetune_retrieval.json").write_text(json.dumps({
        "config": vars(args), "n_train_queries": len(tr_q),
        "n_train_pairs": len(pairs), "n_val_queries": len(va_q),
        "corpus_size": len(cids), "trainable_params": n_tr,
        "loss_history": hist,
        "base": {"train": base_tr, "val": base_va},
        "finetuned": {"train": ft_tr, "val": ft_va},
    }, indent=2, default=float))
    print(f"\nwrote {outdir / 'lfm25_finetune_retrieval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
