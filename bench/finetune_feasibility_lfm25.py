"""Measure whether fine-tuning an LFM2.5 encoder is feasible on this box.

Times one real forward+backward+step at realistic settings, for three configs:
full fine-tune, LoRA, and LoRA with a frozen backbone in eval mode. Then
extrapolates to the cookbook's recipe (3 epochs) so the answer is a wall-clock
number rather than a vibe.

Uses LFM2.5-Embedding-350M as a stand-in for Encoder-350M: same lfm2 architecture,
same 16 layers / 1024 hidden / 354M params, and it is already in the local HF
cache. Step time is a property of the architecture and shapes, not the weights,
so the timing transfers; only the task head differs.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

MODEL = "LiquidAI/LFM2.5-Embedding-350M"


class Classifier(nn.Module):
    """Mean-pool + linear head, matching the cookbook's shape."""

    def __init__(self, backbone, hidden, n_labels):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(pooled)


def build(mode, n_labels, grad_ckpt):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32)
    hidden = backbone.config.hidden_size

    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        # Target the attention projections; conv layers are not Linear so PEFT
        # cannot wrap them, which matters here -- 10 of 16 LFM2 layers are conv.
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        backbone = get_peft_model(backbone, cfg)
    elif mode == "frozen":
        for p in backbone.parameters():
            p.requires_grad = False

    if grad_ckpt and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    model = Classifier(backbone, hidden, n_labels)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def bench(mode, *, batch, seqlen, steps, n_labels, grad_ckpt):
    torch.set_num_threads(4)
    model, trainable, total = build(mode, n_labels, grad_ckpt)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-5)
    lossf = nn.BCEWithLogitsLoss()

    ids = torch.randint(0, 60000, (batch, seqlen))
    mask = torch.ones_like(ids)
    y = torch.zeros(batch, n_labels)
    y[:, 0] = 1.0

    for _ in range(1):  # warmup
        opt.zero_grad(); lossf(model(ids, mask), y).backward(); opt.step()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(ids, mask), y).backward()
        opt.step()
    per_step = (time.time() - t0) / steps

    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return {"mode": mode, "per_step_s": per_step, "trainable": trainable,
            "total": total, "peak_rss_gb": rss,
            "examples_per_s": batch / per_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--labels", type=int, default=10)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--modes", nargs="*", default=["full", "lora", "frozen"])
    ap.add_argument("--dataset-size", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    print(f"batch={args.batch} seqlen={args.seqlen} labels={args.labels} "
          f"grad_ckpt={args.grad_ckpt} threads=4 device=cpu\n")
    print(f"{'mode':<8} {'trainable':>12} {'%':>6} {'s/step':>8} {'ex/s':>7} "
          f"{'peak RSS':>9}  {'3 epochs x 5k'}")
    print("-" * 78)
    rows = []
    for mode in args.modes:
        r = bench(mode, batch=args.batch, seqlen=args.seqlen, steps=args.steps,
                  n_labels=args.labels, grad_ckpt=args.grad_ckpt)
        total_s = (args.dataset_size / r["examples_per_s"]) * args.epochs
        pct = 100.0 * r["trainable"] / r["total"]
        print(f"{r['mode']:<8} {r['trainable']:>12,} {pct:>5.1f}% "
              f"{r['per_step_s']:>8.2f} {r['examples_per_s']:>7.2f} "
              f"{r['peak_rss_gb']:>8.1f}G  {total_s/3600:>6.1f} h")
        r["hours_3_epochs_5k"] = round(total_s / 3600, 2)
        r["trainable_pct"] = round(pct, 4)
        rows.append(r)

    import json, pathlib as _p
    out = _p.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "lfm25_finetune_feasibility.json").write_text(json.dumps(
        {"batch": args.batch, "seqlen": args.seqlen, "labels": args.labels,
         "device": "cpu", "threads": 4, "grad_checkpointing": args.grad_ckpt,
         "rows": rows}, indent=2))
    print(f"\nwrote {out / 'lfm25_finetune_feasibility.json'}")


if __name__ == "__main__":
    main()
