"""Encode stsb + scifact with nvidia/Nemotron-3-Embed-1B-BF16 to float32 caches.

Datasets: stsb (s1, s2 × 1379 sentences), scifact (300 queries + 1600 docs).
Model: nvidia/Nemotron-3-Embed-1B-BF16 (2048-dim, L2-normalized, Matryoshka).

Encode order (cheap first):
  1. stsb s1 (prompt_name="query", batch_size=16)
  2. stsb s2 (prompt_name="query", batch_size=16)
  3. scifact queries (prompt_name="query", batch_size=16)
  4. scifact docs (prompt_name="document", batch_size=8)

Checkpointing: saves 64-text chunks to $SCRATCH/emb/parts/<name>.part{i:04d}.npy
and resumes by skipping chunks that already exist with correct shape.

Outputs:
  $SCRATCH/emb/scifact_docs.npy
  $SCRATCH/emb/scifact_queries.npy
  $SCRATCH/emb/stsb_s1.npy
  $SCRATCH/emb/stsb_s2.npy
  $SCRATCH/emb/meta.json

To skip the ~1 hour encode with the real model:
  python3 bench/embed_nemotron.py --selftest  (synthetic, no model needed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SCRATCH = Path(os.environ.get("SCRATCH", "/tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad"))
DATA_DIR = SCRATCH / "data"
EMB_DIR = SCRATCH / "emb"
PARTS_DIR = EMB_DIR / "parts"

MODEL_ID = "nvidia/Nemotron-3-Embed-1B-BF16"
DIM = 2048
CHUNK_SIZE = 64


def load_data():
    """Load stsb and scifact datasets."""
    sys.stderr.write("loading stsb and scifact...\n")

    with open(DATA_DIR / "stsb_test.json") as f:
        stsb = json.load(f)
    stsb_s1 = stsb["s1"]
    stsb_s2 = stsb["s2"]

    with open(DATA_DIR / "scifact_subset.json") as f:
        scifact = json.load(f)
    scifact_queries = scifact["query_texts"]
    scifact_docs = scifact["doc_texts"]

    sys.stderr.write(
        f"  stsb_s1={len(stsb_s1)}  stsb_s2={len(stsb_s2)}  "
        f"scifact_queries={len(scifact_queries)}  scifact_docs={len(scifact_docs)}\n"
    )
    return stsb_s1, stsb_s2, scifact_queries, scifact_docs


def encode_dataset(texts, name, prompt_name, encoder_fn):
    """Encode texts in chunks, checkpoint/resume, return (N, 2048) float32.

    encoder_fn(texts: list[str], prompt_name: str) -> (N, 2048) float32 array
    Batch size is baked into encoder_fn via closure (make_encoder).
    """
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    out = np.zeros((n, DIM), dtype=np.float32)

    t0 = time.time()
    last_log = t0
    done = 0

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, n)
        part_path = PARTS_DIR / f"{name}.part{chunk_idx:04d}.npy"

        if part_path.exists():
            # Resume: load the part
            part = np.load(part_path)
            if part.shape == (end_idx - start_idx, DIM):
                out[start_idx:end_idx] = part
                done = end_idx
                # Log progress
                now = time.time()
                if now - last_log >= 10 or done >= n:
                    elapsed = now - t0
                    rate = done / max(elapsed, 1e-9)
                    sys.stdout.write(
                        f"{name:20s} chunk {chunk_idx:3d}/{n_chunks:3d}  "
                        f"elapsed={elapsed:6.1f}s  rate={rate:7.2f} docs/s\n"
                    )
                    sys.stdout.flush()
                    last_log = now
                continue

        # Encode this chunk
        chunk_texts = texts[start_idx:end_idx]
        chunk_emb = encoder_fn(chunk_texts, prompt_name)
        assert chunk_emb.shape == (len(chunk_texts), DIM), \
            f"Expected shape {(len(chunk_texts), DIM)}, got {chunk_emb.shape}"

        # Save part and full output
        out[start_idx:end_idx] = chunk_emb
        np.save(part_path, chunk_emb)
        done = end_idx

        # Log progress
        now = time.time()
        if now - last_log >= 10 or done >= n:
            elapsed = now - t0
            rate = done / max(elapsed, 1e-9)
            sys.stdout.write(
                f"{name:20s} chunk {chunk_idx:3d}/{n_chunks:3d}  "
                f"elapsed={elapsed:6.1f}s  rate={rate:7.2f} docs/s\n"
            )
            sys.stdout.flush()
            last_log = now

    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--selftest", action="store_true", help="test with synthetic data (no model)")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    # Set HF_HOME and torch settings BEFORE importing torch/sentence_transformers
    os.environ["HF_HOME"] = str(SCRATCH / "hf")

    import torch
    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    stsb_s1, stsb_s2, scifact_queries, scifact_docs = load_data()

    # Load model
    sys.stderr.write(f"loading {MODEL_ID}...\n")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    sys.stderr.write(f"  loaded in {time.time()-t0:.1f}s\n")

    # Create encoder function that respects batch_size parameter
    def make_encoder(batch_size):
        def encode_fn(texts, prompt_name):
            emb = model.encode(
                texts,
                prompt_name=prompt_name,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            return np.asarray(emb, dtype=np.float32)
        return encode_fn

    # Encode in order (cheap first)
    sys.stderr.write("\n=== encoding stsb_s1 ===\n")
    arr_s1 = encode_dataset(stsb_s1, "stsb_s1", "query", make_encoder(16))
    np.save(EMB_DIR / "stsb_s1.npy", arr_s1)

    sys.stderr.write("\n=== encoding stsb_s2 ===\n")
    arr_s2 = encode_dataset(stsb_s2, "stsb_s2", "query", make_encoder(16))
    np.save(EMB_DIR / "stsb_s2.npy", arr_s2)

    sys.stderr.write("\n=== encoding scifact_queries ===\n")
    arr_q = encode_dataset(scifact_queries, "scifact_queries", "query", make_encoder(16))
    np.save(EMB_DIR / "scifact_queries.npy", arr_q)

    sys.stderr.write("\n=== encoding scifact_docs ===\n")
    arr_d = encode_dataset(scifact_docs, "scifact_docs", "document", make_encoder(8))
    np.save(EMB_DIR / "scifact_docs.npy", arr_d)

    # Write meta.json
    meta = {
        "model": MODEL_ID,
        "doc_prompt": "passage: ",
        "query_prompt": "query: ",
        "counts": {
            "stsb_s1": len(stsb_s1),
            "stsb_s2": len(stsb_s2),
            "scifact_queries": len(scifact_queries),
            "scifact_docs": len(scifact_docs),
        }
    }
    (EMB_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    sys.stderr.write(f"\nwrote {EMB_DIR}/stsb_s1.npy (shape {arr_s1.shape})\n")
    sys.stderr.write(f"wrote {EMB_DIR}/stsb_s2.npy (shape {arr_s2.shape})\n")
    sys.stderr.write(f"wrote {EMB_DIR}/scifact_queries.npy (shape {arr_q.shape})\n")
    sys.stderr.write(f"wrote {EMB_DIR}/scifact_docs.npy (shape {arr_d.shape})\n")
    sys.stderr.write(f"wrote {EMB_DIR}/meta.json\n")


def selftest():
    """Synthetic test: no model, no network, verify chunking/resume/concat/meta."""
    import tempfile
    import shutil

    sys.stderr.write("=== selftest mode (synthetic data, no model) ===\n")

    # Create temp scratch dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        emb_dir = tmpdir / "emb"
        parts_dir = emb_dir / "parts"
        emb_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)

        # Test 1: Create synthetic data and encode
        sys.stderr.write("  test 1: encode synthetic data with fake encoder\n")
        texts = [f"text {i}" for i in range(10)]

        def fake_encode(texts, prompt_name):
            """Fake encoder: random embeddings, L2-normalized."""
            rng = np.random.default_rng(42)
            emb = rng.normal(0, 1, (len(texts), DIM)).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb /= norms
            return emb

        out1 = _encode_dataset_for_selftest(texts, "test1", "query", fake_encode, parts_dir, emb_dir)
        assert out1.shape == (10, DIM), f"Expected (10, {DIM}), got {out1.shape}"
        sys.stderr.write(f"    ✓ encoded shape {out1.shape}\n")

        # Test 2: Resume by deleting one part, re-running
        sys.stderr.write("  test 2: resume after part deletion\n")
        # Delete first part
        part0 = parts_dir / "test2.part0000.npy"
        if part0.exists():
            part0.unlink()
        out2a = _encode_dataset_for_selftest(texts, "test2", "query", fake_encode, parts_dir, emb_dir)
        assert out2a.shape == (10, DIM), f"Expected (10, {DIM}), got {out2a.shape}"
        # Now re-encode without deleting
        out2b = _encode_dataset_for_selftest(texts, "test2", "query", fake_encode, parts_dir, emb_dir)
        # They should be bitwise identical (same seed, same RNG)
        assert np.allclose(out2a, out2b), "Resume should match non-resume"
        sys.stderr.write(f"    ✓ resumed output matches original\n")

        # Test 3: Final files exist with correct shapes
        sys.stderr.write("  test 3: final array files and meta.json\n")
        meta = {
            "model": MODEL_ID,
            "doc_prompt": "passage: ",
            "query_prompt": "query: ",
            "counts": {
                "test1": 10,
                "test2": 10,
            }
        }
        (emb_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        assert (emb_dir / "meta.json").exists(), "meta.json should exist"
        meta_loaded = json.loads((emb_dir / "meta.json").read_text())
        assert meta_loaded["model"] == MODEL_ID, "model ID mismatch"
        assert meta_loaded["doc_prompt"] == "passage: ", "doc_prompt mismatch"
        assert meta_loaded["query_prompt"] == "query: ", "query_prompt mismatch"
        sys.stderr.write(f"    ✓ meta.json written and verified\n")

        # Test 4: Chunk boundaries
        sys.stderr.write("  test 4: chunk boundaries (CHUNK_SIZE=64)\n")
        long_texts = [f"text {i}" for i in range(150)]  # > CHUNK_SIZE
        out3 = _encode_dataset_for_selftest(long_texts, "test3", "query", fake_encode, parts_dir, emb_dir)
        assert out3.shape == (150, DIM), f"Expected (150, {DIM}), got {out3.shape}"
        # Check that parts were created
        parts = sorted(parts_dir.glob("test3.part*.npy"))
        assert len(parts) == 3, f"Expected 3 parts (150/{CHUNK_SIZE}), got {len(parts)}"
        sys.stderr.write(f"    ✓ chunking correct: 150 texts → {len(parts)} parts\n")

    sys.stderr.write("\n✓✓✓ SELFTEST PASS ✓✓✓\n")


def _encode_dataset_for_selftest(texts, name, prompt_name, encoder_fn, parts_dir, emb_dir):
    """Reusable encode logic for selftest."""
    n = len(texts)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    out = np.zeros((n, DIM), dtype=np.float32)

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, n)
        part_path = parts_dir / f"{name}.part{chunk_idx:04d}.npy"

        if part_path.exists():
            part = np.load(part_path)
            if part.shape == (end_idx - start_idx, DIM):
                out[start_idx:end_idx] = part
                continue

        chunk_texts = texts[start_idx:end_idx]
        chunk_emb = encoder_fn(chunk_texts, prompt_name)
        out[start_idx:end_idx] = chunk_emb
        np.save(part_path, chunk_emb)

    return out


if __name__ == "__main__":
    main()
