"""Encode a BEIR-style corpus with LFM2.5-Embedding-350M.

Produces the .npy files bench/eval_lfm25.py consumes. Mirrors embed_nemotron.py:
CPU-only, checkpointed, resumable, and it writes the exact strings it embedded so
a cache can be audited after the fact.

Model facts pinned from the HF repo (do not "simplify" these away):
  * CLS pooling  -- 1_Pooling/config.json sets pooling_mode_cls_token=true
  * prompts      -- "query: " / "document: " from config_sentence_transformers.json.
                    The card is explicit that omitting them silently degrades
                    retrieval, so they are applied here, not left to the caller.
  * max_seq_len  -- 512 per sentence_bert_config.json
  * L2 normalize -- similarity_fn_name is cosine

    python bench/embed_lfm25.py --data-dir <scifact json dir> \
                                --out bench/.cache/LFM25_SCIFACT
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

MODEL_ID = "LiquidAI/LFM2.5-Embedding-350M"
MAX_SEQ = 512
PROMPTS = {"query": "query: ", "document": "document: "}


def load_model(model_id: str, threads: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    return tok, model


def encode(tok, model, texts: list[str], *, prompt: str, batch_size: int,
           progress_every: int = 10) -> np.ndarray:
    """Encode with CLS pooling + L2 norm, in length-sorted batches.

    Padding is BatchLongest, so mixing a 40-token abstract with a 512-token one
    makes the short rows cost as much as the long one. Sorting by length first
    and scattering back at the end is worth ~2x on a corpus with SciFact's
    length spread, and is numerically identical -- attention_mask already makes
    padding a no-op, so the only thing that changes is how much padding exists.
    """
    import torch

    prefix = PROMPTS[prompt]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for bi, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start : start + batch_size]
            chunk = [prefix + texts[i] for i in idx]
            enc = tok(
                chunk, padding=True, truncation=True,
                max_length=MAX_SEQ, return_tensors="pt",
            )
            hidden = model(**enc).last_hidden_state
            vec = hidden[:, 0]  # CLS pooling -- 1_Pooling/config.json
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out[idx] = vec.float().numpy()
            if bi % progress_every == 0:
                done = start + len(idx)
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(texts) - done) / max(rate, 1e-9)
                print(
                    f"    {done}/{len(texts)}  {rate:.1f}/s  eta {eta/60:.1f}m",
                    flush=True,
                )
    return out


def _run_split(name, records, tok, model, outdir, batch_size, prompt):
    vec_path = outdir / f"{name}.npy"
    id_path = outdir / f"{name[:-1] if name.endswith('s') else name}_ids.json"
    id_path = outdir / ("doc_ids.json" if name == "docs" else "query_ids.json")
    if vec_path.exists():
        print(f"  {name}: cached, skipping")
        return
    print(f"  {name}: encoding {len(records)} texts (prompt={prompt!r})")
    t0 = time.time()
    vecs = encode(tok, model, [r["text"] for r in records],
                  prompt=prompt, batch_size=batch_size)
    np.save(vec_path, vecs)
    id_path.write_text(json.dumps([r["id"] for r in records]))
    print(f"  {name}: {vecs.shape} in {(time.time()-t0)/60:.1f}m -> {vec_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N docs")
    args = ap.parse_args()

    data = pathlib.Path(args.data_dir)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    docs = json.loads((data / "docs.json").read_text())
    queries = json.loads((data / "queries.json").read_text())
    if args.limit:
        docs = docs[: args.limit]
        queries = queries[: max(4, args.limit // 10)]

    print(f"loading {args.model} ...")
    t0 = time.time()
    tok, model = load_model(args.model, args.threads)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time()-t0:.1f}s  params={n_params/1e6:.1f}M  "
          f"hidden={model.config.hidden_size}")

    _run_split("queries", queries, tok, model, outdir, args.batch_size, "query")
    _run_split("docs", docs, tok, model, outdir, args.batch_size, "document")

    (outdir / "meta.json").write_text(json.dumps({
        "model_id": args.model, "pooling": "cls", "prompts": PROMPTS,
        "max_seq_length": MAX_SEQ, "normalize_l2": True,
        "n_docs": len(docs), "n_queries": len(queries),
        "hidden_size": int(model.config.hidden_size),
    }, indent=2))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
