"""Smoke test: load jina-embeddings-v5-text-nano on CPU, time a small batch.

Goal: measure tokens/sec and tokens/abstract on real SPECTER2 texts so we can
project the full 10k * 4-adapter run before kicking it off.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
CACHE = Path(__file__).resolve().parent / ".cache" / "SPECTER2"

# Limit CPU torch threads to avoid thrashing.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

texts = json.loads((CACHE / "texts.json").read_text())[:50]
print(f"loaded {len(texts)} abstracts; median chars={int(np.median([len(t) for t in texts]))}")

t0 = time.time()
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float32,
)
model.eval()
print(f"loaded model in {time.time()-t0:.1f}s; params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Try each adapter with a tiny batch
TASKS = ["retrieval", "text-matching", "clustering", "classification"]
for task in TASKS:
    t0 = time.time()
    emb = model.encode(texts=texts[:8], task=task, prompt_name="document")
    dt = time.time() - t0
    arr = emb.cpu().float().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
    print(f"task={task:16s}  shape={arr.shape}  norms={np.linalg.norm(arr, axis=1)[:3].round(3).tolist()}  {dt:.1f}s for 8 ({dt/8*1000:.0f}ms/text)")

# Time a bigger batch on retrieval — batched manually since encode() has no batch arg.
t0 = time.time()
out = []
B = 8
for i in range(0, len(texts), B):
    e = model.encode(texts=texts[i:i+B], task="retrieval", prompt_name="document")
    out.append(e.cpu().float().numpy() if hasattr(e, "cpu") else np.asarray(e))
arr = np.concatenate(out)
dt = time.time() - t0
print(f"\n{len(texts)} texts at retrieval (batch={B}): {dt:.1f}s ({dt/len(texts)*1000:.0f}ms/text)")
print(f"projected 10000 texts × 1 adapter: {dt/len(texts)*10000/60:.1f} min")
print(f"projected 10000 texts × 4 adapters: {dt/len(texts)*10000*4/60:.1f} min")
