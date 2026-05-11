"""Encode SPECTER2 title+abstract texts with jina-embeddings-v5-text-nano.

Output: bench/.cache/JINA_V5_NANO/{task}.npy  shape (N, 768) float32.

Per-task .npy files let us slice matryoshka dims offline (sign-bit then
re-normalize is mathematically equivalent to slicing the un-normalized
pooled vector then normalizing — see modeling_jina_embeddings_v5.py:110-112).

Resumable: skips a task if its .npy is already on disk with the expected
shape. Re-run with --overwrite to force.

Usage:
    python bench/embed_jina.py --n 2500 --batch 16 --max-len 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
TASKS = ("retrieval", "text-matching", "clustering", "classification")
DIM = 768

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "bench" / ".cache" / "SPECTER2" / "texts.json"
DST_DIR = REPO_ROOT / "bench" / ".cache" / "JINA_V5_NANO"


def encode_task(model, texts, task, batch_size, max_length):
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    last_log = t0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        emb = model.encode(
            texts=chunk,
            task=task,
            prompt_name="document",
            max_length=max_length,
        )
        arr = emb.detach().cpu().float().numpy()
        out[i : i + len(chunk)] = arr
        now = time.time()
        if now - last_log > 30 or i + batch_size >= len(texts):
            done = i + len(chunk)
            rate = done / (now - t0)
            eta = (len(texts) - done) / max(rate, 1e-9)
            sys.stderr.write(
                f"  [{task}] {done}/{len(texts)}  "
                f"rate={rate:.2f}/s  eta={eta/60:.1f}min\n"
            )
            sys.stderr.flush()
            last_log = now
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    texts = json.loads(SRC.read_text())[: args.n]
    sys.stderr.write(f"loaded {len(texts)} texts from {SRC}\n")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": MODEL_ID,
        "n": len(texts),
        "dim": DIM,
        "max_length": args.max_len,
        "batch_size": args.batch,
        "source": "SPECTER2 title [SEP] abstract",
        "prompt_name": "document",
        "tasks": list(args.tasks),
    }
    (DST_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    t_model = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    sys.stderr.write(f"loaded model in {time.time()-t_model:.1f}s\n")

    for task in args.tasks:
        dst = DST_DIR / f"{task}.npy"
        if dst.exists() and not args.overwrite:
            arr = np.load(dst, mmap_mode="r")
            if arr.shape == (len(texts), DIM):
                sys.stderr.write(f"[skip] {task}: cached at {dst}\n")
                continue
            sys.stderr.write(f"[redo] {task}: cached shape {arr.shape} != expected ({len(texts)}, {DIM})\n")
        sys.stderr.write(f"\n=== encoding task={task} ===\n")
        t0 = time.time()
        emb = encode_task(model, texts, task, args.batch, args.max_len)
        # All rows are L2-normalized by encode(). Sanity check.
        norms = np.linalg.norm(emb, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            sys.stderr.write(f"  WARN: norms range {norms.min():.4f}..{norms.max():.4f}\n")
        np.save(dst, emb)
        sys.stderr.write(f"  wrote {dst} in {(time.time()-t0)/60:.1f}min\n")

    sys.stderr.write("\nall done.\n")


if __name__ == "__main__":
    main()
