"""Multi-label classification on FROZEN LFM2.5 embeddings — the cookbook, cheaply.

Liquid's cookbook example (Liquid4All/cookbook, examples/lfm-encoder-classification)
full-fine-tunes LFM2.5-Encoder-350M with the HF Trainer: mean pooling, a linear
head, 3 epochs, lr 2e-5, fp32, GPU assumed. Measured on a 4-core CPU box, that
recipe costs 15-18 h for 3 epochs over 5k documents (two runs: 14.8 h and
18.4 h -- CPU timing varies, the ~6x ratio to the frozen path does not), at
13.7 GB peak RSS against a 15 GB ceiling.

This is the same pipeline with the backbone frozen and its output cached:

    encode once  ->  train a head on cached vectors  ->  tune thresholds  ->  eval

Measured on the same box: the encode is ~83 min for 5,183 documents at 512
tokens, and every subsequent step is sub-second. So iterating on head
architecture, class weighting or thresholds is free, where the cookbook's recipe
pays the full forward+backward again each time.

What you give up is real: the encoder cannot adapt its representation to your
domain. Freeze-and-cache is the right default when the domain is close to
pretraining and the label set is what is new. It is the wrong choice when the
text itself is far out of distribution -- that is what the GPU recipe is for.

    # tiny synthetic demo, runs end to end with no data
    python bench/frozen_classifier_lfm25.py --demo

    # real use
    python bench/frozen_classifier_lfm25.py --train train.jsonl --test test.jsonl
    # jsonl rows: {"text": "...", "labels": ["billing", "urgent"]}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# ---------------------------------------------------------------- data


def load_jsonl(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    return [r["text"] for r in rows], [r.get("labels", []) for r in rows]


def synth(n, seed=0):
    """Label-correlated text so the demo exercises a real signal, not noise."""
    rng = np.random.default_rng(seed)
    topics = {
        "billing": ["invoice", "refund", "charged twice", "payment failed", "subscription"],
        "auth": ["cannot log in", "password reset", "two-factor", "locked out", "sso"],
        "perf": ["very slow", "timeout", "latency spike", "hangs", "unresponsive"],
        "urgent": ["production down", "asap", "critical", "blocking release", "escalate"],
    }
    names = list(topics)
    texts, labels = [], []
    for _ in range(n):
        k = rng.integers(1, 3)
        picked = list(rng.choice(names, size=k, replace=False))
        frag = [rng.choice(topics[t]) for t in picked for _ in range(2)]
        rng.shuffle(frag)
        texts.append("Ticket: " + ", ".join(frag) + ".")
        labels.append(picked)
    return texts, labels


def binarize(label_lists, classes=None):
    # str(): rng.choice yields np.str_, which prints as np.str_('auth')
    classes = classes or sorted({str(l) for ls in label_lists for l in ls})
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(label_lists), len(classes)), dtype=np.float32)
    for r, ls in enumerate(label_lists):
        for l in ls:
            if str(l) in idx:
                Y[r, idx[str(l)]] = 1.0
    return Y, classes


# ---------------------------------------------------------------- model


def encode(texts, model_id, batch_size, cache=None):
    """Cached forward pass. This is the only expensive step, and it runs once."""
    if cache and pathlib.Path(cache).exists():
        print(f"  cache hit: {cache}")
        return np.load(cache)
    from embed_lfm25 import load_model, encode as _enc
    tok, model = load_model(model_id, 4)
    t0 = time.time()
    V = _enc(tok, model, texts, prompt="document", batch_size=batch_size)
    print(f"  encoded {len(texts)} texts in {(time.time() - t0) / 60:.1f} min")
    if cache:
        pathlib.Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, V)
    return V


def train_head(X, Y, *, epochs, lr, batch, seed, hidden=0):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    layers = ([torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
               torch.nn.Linear(hidden, Y.shape[1])] if hidden
              else [torch.nn.Linear(X.shape[1], Y.shape[1])])
    head = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    # Same loss as the cookbook: BCE-with-logits over independent labels.
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lossf(head(Xt[b]), Yt[b]).backward()
            opt.step()
    secs = time.time() - t0
    with torch.no_grad():
        logits = head(Xt)
    return head, float(secs), torch.sigmoid(logits).numpy()


def predict(head, X):
    import torch
    with torch.no_grad():
        return torch.sigmoid(head(torch.from_numpy(X))).numpy()


# ---------------------------------------------------------------- metrics


def f1(y, p):
    tp = float((y * p).sum())
    if tp == 0:
        return 0.0
    # float() throughout: numpy scalars leak into the results dict otherwise and
    # json.dumps rejects np.float32 at the very end of the run.
    prec = tp / max(float(p.sum()), 1e-9)
    rec = tp / max(float(y.sum()), 1e-9)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


def tune_thresholds(Y, P, grid=None):
    """Per-label threshold sweep — the cookbook's validation stage.

    Free here: it is a scan over cached probabilities, no model involved. On the
    fine-tuning path the same stage needs a full validation forward pass.
    """
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.01)
    return np.array([max(grid, key=lambda t: f1(Y[:, c], (P[:, c] >= t).astype(np.float32)))
                     for c in range(Y.shape[1])], dtype=np.float32)


def report(Y, P, thr, classes, label):
    pred = (P >= thr[None, :]).astype(np.float32)
    micro = f1(Y, pred)
    macro = float(np.mean([f1(Y[:, c], pred[:, c]) for c in range(Y.shape[1])]))
    print(f"\n  {label}:  micro-F1 {micro:.4f}   macro-F1 {macro:.4f}")
    for c, name in enumerate(classes):
        print(f"    {name:<12} thr {thr[c]:.2f}  F1 {f1(Y[:, c], pred[:, c]):.4f}  "
              f"support {int(Y[:, c].sum())}")
    return {"micro_f1": micro, "macro_f1": macro}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-Embedding-350M")
    ap.add_argument("--cache-dir", default="bench/.cache/CLF")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    if args.demo:
        tr_x, tr_y = synth(600, seed=args.seed)
        te_x, te_y = synth(200, seed=args.seed + 1)
        print(f"demo: {len(tr_x)} train / {len(te_x)} test synthetic tickets")
    elif args.train and args.test:
        tr_x, tr_y = load_jsonl(args.train)
        te_x, te_y = load_jsonl(args.test)
    else:
        ap.error("pass --demo, or both --train and --test")

    Ytr, classes = binarize(tr_y)
    Yte, _ = binarize(te_y, classes)
    print(f"labels: {classes}")

    cd = pathlib.Path(args.cache_dir)
    print("\nencoding (the only expensive step, and it runs once):")
    Xtr = encode(tr_x, args.model, args.encode_batch, cd / "train.npy")
    Xte = encode(te_x, args.model, args.encode_batch, cd / "test.npy")

    head, secs, Ptr = train_head(Xtr, Ytr, epochs=args.epochs, lr=args.lr,
                                 batch=args.batch, seed=args.seed,
                                 hidden=args.hidden)
    print(f"\nhead trained on cached vectors: {args.epochs} epochs in {secs:.2f} s")

    thr = tune_thresholds(Ytr, Ptr)
    Pte = predict(head, Xte)
    half = np.full(len(classes), 0.5, dtype=np.float32)
    m_fixed = report(Yte, Pte, half, classes, "test @ fixed 0.5")
    m_tuned = report(Yte, Pte, thr, classes, "test @ tuned thresholds")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lfm25_frozen_classifier.json").write_text(json.dumps({
        "classes": classes, "n_train": len(tr_x), "n_test": len(te_x),
        "head_train_seconds": round(secs, 3), "epochs": args.epochs,
        "hidden": args.hidden, "thresholds": thr.tolist(),
        "fixed_0.5": m_fixed, "tuned": m_tuned,
    }, indent=2))
    print(f"\nwrote {outdir / 'lfm25_frozen_classifier.json'}")
    print(f"\nhead training cost {secs:.2f} s. The cookbook's full fine-tune of "
          f"the same shape\nmeasured 15-18 h for 3 epochs over 5k docs on this box "
          f"(bench/results/lfm25_finetune_feasibility.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
