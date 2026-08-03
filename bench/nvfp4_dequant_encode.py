"""Run NVIDIA's NVFP4 Nemotron-3-Embed-1B on CPU (no GPU) and encode the eval sets.

The published `nvidia/Nemotron-3-Embed-1B-NVFP4` checkpoint stores linear weights
in NVFP4 (4-bit e2m1 values, packed 2-per-byte as uint8, with a per-16-element
FP8-e4m3 block scale and a per-tensor FP32 global scale). Inference is documented
for GPU + vLLM only. But NVFP4 is a *weight* encoding: the dequantized weights are
exactly the values the GPU kernel multiplies with. So we can reconstruct them on
CPU and run the model anywhere.

    W[o, i] = e2m1(nibble[o, i]) * weight_scale[o, i // 16] * weight_scale_2

`build` reconstructs a plain float32 checkpoint from the NVFP4 one (validated:
per-weight cosine ≈ 0.996 vs the BF16 twin, i.e. 4-bit quant noise and nothing
else). `encode` runs it over SciFact + STS-B, chunked and resumable, to
`$SCRATCH/emb_nvfp4/`.

Fidelity note: this reconstructs the NVFP4 *weights* faithfully but does NOT
simulate NVFP4 *activation* quantization (the checkpoint's static `input_scale`
per linear). Activations run in float32, so these embeddings are a mild
upper bound on NVFP4 quality — the weight-quantization error, which dominates,
is captured exactly. NVIDIA's own BF16->NVFP4 RTEB delta is 0.38 nDCG@10.

Usage
-----
    python bench/nvfp4_dequant_encode.py build     # NVFP4 checkpoint -> fp32 local dir
    python bench/nvfp4_dequant_encode.py encode     # -> $SCRATCH/emb_nvfp4/*.npy
    python bench/nvfp4_dequant_encode.py --selftest # offline e2m1/dequant unit checks
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Shared resolution — see bench/nemotron_paths.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemotron_paths import DATA_DIR, SCRATCH  # noqa: E402
EMB_DIR = Path(os.environ.get("NVFP4_EMB_DIR", SCRATCH / "emb_nvfp4"))
DEQUANT_DIR = Path(os.environ.get("NVFP4_DEQUANT_DIR", SCRATCH / "nvfp4_dequant"))
NVFP4_MODEL = "nvidia/Nemotron-3-Embed-1B-NVFP4"
DIM = 2048
CHUNK = 64

# FP4 e2m1 magnitude table, indexed by the low 3 bits of a nibble; the high bit
# is the sign. Values: {0, .5, 1, 1.5, 2, 3, 4, 6}.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_decode(nibbles: np.ndarray) -> np.ndarray:
    """Decode uint4 nibbles (0..15) to their FP4 e2m1 float values."""
    nibbles = nibbles.astype(np.uint8)
    sign = np.where(nibbles >= 8, -1.0, 1.0).astype(np.float32)
    return sign * E2M1_MAG[nibbles & 7]


def dequant_linear(packed, weight_scale, weight_scale_2):
    """Reconstruct one NVFP4 linear weight to a dense float32 matrix.

    packed        : uint8 (out, in/2) — two e2m1 nibbles per byte, element 2i in
                    the low nibble, 2i+1 in the high nibble.
    weight_scale  : float (out, in/16) — per-16-element block scale (FP8 e4m3).
    weight_scale_2: scalar float — per-tensor global scale.
    """
    packed = packed.astype(np.uint8)
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    out, in_half = packed.shape
    fp4 = np.empty((out, in_half * 2), dtype=np.float32)
    fp4[:, 0::2] = e2m1_decode(lo)
    fp4[:, 1::2] = e2m1_decode(hi)
    block = np.repeat(np.asarray(weight_scale, dtype=np.float32), 16, axis=1)
    return fp4 * block * float(weight_scale_2)


def build() -> None:
    """Reconstruct a float32 checkpoint from the NVFP4 one."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", str(SCRATCH / "hf"))
    nv_dir = snapshot_download(NVFP4_MODEL)
    DEQUANT_DIR.mkdir(parents=True, exist_ok=True)

    # copy everything except the safetensors (configs, tokenizer, ST modules)
    for f in os.listdir(nv_dir):
        if f.endswith(".safetensors"):
            continue
        src = os.path.join(nv_dir, f)
        dst = DEQUANT_DIR / f
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # strip the quantization metadata so it loads as a plain model
    cfg = json.load(open(DEQUANT_DIR / "config.json"))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(DEQUANT_DIR / "config.json", "w"), indent=2)
    for junk in ("hf_quant_config.json", "quantization_metadata.json"):
        p = DEQUANT_DIR / junk
        if p.exists():
            p.unlink()

    sf = glob.glob(os.path.join(nv_dir, "*.safetensors"))[0]
    st = safe_open(sf, "pt")
    keys = set(st.keys())
    bases = {k[: -len("weight_scale")] for k in keys if k.endswith(".weight_scale")}
    new, nq, nc = {}, 0, 0
    for k in keys:
        if k.endswith(("weight_scale", "weight_scale_2", "input_scale")):
            continue
        base = k[: -len("weight")] if k.endswith("weight") else None
        if base is not None and base in bases:
            W = dequant_linear(
                st.get_tensor(base + "weight").numpy(),
                st.get_tensor(base + "weight_scale").float().numpy(),
                st.get_tensor(base + "weight_scale_2"),
            )
            new[k] = torch.from_numpy(W).to(torch.float32)
            nq += 1
        else:
            new[k] = st.get_tensor(k).to(torch.float32)
            nc += 1
    save_file(new, str(DEQUANT_DIR / "model.safetensors"), metadata={"format": "pt"})
    size_gb = (DEQUANT_DIR / "model.safetensors").stat().st_size / 1e9
    print(f"built {DEQUANT_DIR}: {nq} dequantized linears, {nc} copied ({size_gb:.2f} GB)")


def _encode_all() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HOME", str(SCRATCH / "hf"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    parts = EMB_DIR / "parts"
    parts.mkdir(exist_ok=True)

    if not (DEQUANT_DIR / "model.safetensors").exists():
        sys.exit(f"dequant checkpoint missing at {DEQUANT_DIR}; run `build` first.")

    scifact = json.load(open(DATA_DIR / "scifact_subset.json"))
    stsb = json.load(open(DATA_DIR / "stsb_test.json"))
    jobs = [
        ("stsb_s1", stsb["s1"], "query", 16),
        ("stsb_s2", stsb["s2"], "query", 16),
        ("scifact_queries", scifact["query_texts"], "query", 16),
        ("scifact_docs", scifact["doc_texts"], "document", 8),
    ]

    print(f"loading dequant model from {DEQUANT_DIR} ...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(
        str(DEQUANT_DIR), device="cpu", model_kwargs={"torch_dtype": torch.float32}
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    for name, texts, prompt, bs in jobs:
        final = EMB_DIR / f"{name}.npy"
        if final.exists() and np.load(final, mmap_mode="r").shape == (len(texts), DIM):
            print(f"{name}: already done", flush=True)
            continue
        n = len(texts)
        nch = (n + CHUNK - 1) // CHUNK
        out = np.zeros((n, DIM), dtype=np.float32)
        t0 = time.time()
        for c in range(nch):
            s, e = c * CHUNK, min((c + 1) * CHUNK, n)
            pp = parts / f"{name}.part{c:04d}.npy"
            if pp.exists() and np.load(pp).shape == (e - s, DIM):
                out[s:e] = np.load(pp)
                continue
            emb = model.encode(
                texts[s:e], prompt_name=prompt, normalize_embeddings=True, batch_size=bs
            ).astype(np.float32)
            out[s:e] = emb
            np.save(pp, emb)
            el = time.time() - t0
            print(f"{name} chunk {c+1}/{nch} elapsed={el:6.1f}s rate={e/max(el,1e-9):.2f}/s", flush=True)
        np.save(final, out)
        print(f"wrote {final} {out.shape}", flush=True)

    meta = {
        "model": NVFP4_MODEL,
        "reconstruction": "NVFP4 weights dequantized to float32 on CPU; "
        "activation quant NOT simulated (float32 activations)",
        "query_prompt": "query: ",
        "doc_prompt": "passage: ",
        "counts": {name: len(t) for name, t, _, _ in jobs},
    }
    (EMB_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {EMB_DIR/'meta.json'}", flush=True)


def selftest() -> None:
    print("[nvfp4] selftest ...")
    ok = 0
    # e2m1 endpoints
    vals = e2m1_decode(np.array([0, 1, 2, 4, 7, 8, 9, 15]))
    exp = np.array([0.0, 0.5, 1.0, 2.0, 6.0, -0.0, -0.5, -6.0], dtype=np.float32)
    assert np.allclose(vals, exp), (vals, exp)
    print("PASS: e2m1 decode endpoints"); ok += 1
    # round-trip a hand-built 16-wide block (one block of scale). First two
    # bytes hold nibbles lo,hi = 1,2 then 3,4; the rest are zero.
    packed = np.zeros((1, 8), dtype=np.uint8)  # 8 bytes -> 16 nibbles -> in=16
    packed[0, 0] = 0x21  # low nibble 1, high nibble 2
    packed[0, 1] = 0x43  # low nibble 3, high nibble 4
    W = dequant_linear(packed, np.ones((1, 1), np.float32), np.float32(1.0))
    assert W.shape == (1, 16)
    assert np.allclose(W[0, :4], [0.5, 1.0, 1.5, 2.0]), W[0, :4]
    print("PASS: packed nibble order + block-scale expand"); ok += 1
    # scale application (block scale 2 x global scale 3 = 6)
    W2 = dequant_linear(packed, np.full((1, 1), 2.0, np.float32), np.float32(3.0))
    assert np.allclose(W2[0, :4], np.array([0.5, 1.0, 1.5, 2.0]) * 6.0), W2[0, :4]
    print("PASS: block scale x global scale"); ok += 1
    print(f"[nvfp4] selftest: {ok}/3 passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", choices=["build", "encode"], help="build fp32 ckpt or encode datasets")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.mode == "build":
        build()
    elif args.mode == "encode":
        _encode_all()
    else:
        ap.error("give a mode: build | encode  (or --selftest)")


if __name__ == "__main__":
    main()
