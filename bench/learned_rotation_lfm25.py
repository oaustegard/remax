"""Can a learned rotation beat a random one at the same bits? (post-hoc, no model)

remax rotates with a Haar-random orthogonal matrix before taking signs. The
rotation is data-oblivious, which is the point -- it is reproducible from
(d, seed) and needs no corpus. But it is also, by construction, not tuned to the
data it is about to destroy information about.

This asks what a data-DEPENDENT rotation buys, at identical storage, using only
cached fp32 embeddings. No model weights are touched, so it runs in minutes on
CPU rather than the 15-18 h a real fine-tune of the encoder would cost here.

Three transforms, all producing exactly the same 128 B/vector index:

  haar   Haar-random orthogonal. The current remax default; the baseline.
  itq    Iterative Quantization (Gong & Lazebnik, CVPR 2011). Alternates
         B = sign(XR) with an orthogonal Procrustes update of R, minimizing
         the quantization error ||B - XR||_F. Closed form, unsupervised.
  ste    Straight-through-estimator training of a free linear map, distilling
         the fp32 *ranking* into the binary code with a triplet margin loss.
         This is the "quantization-aware" option: sign() is in the graph and
         gradients flow through it.

EVALUATION INTEGRITY. Every transform is fitted on DOCUMENT embeddings only.
Queries and qrels are never seen during fitting, so the qrels evaluation is
held-out with respect to the thing being learned. --holdout additionally fits on
a random half of the documents to check the transform is learning corpus
geometry rather than memorizing the specific vectors it is scored on.

The transfer risk is real and is the thing most likely to sink this: SciFact
queries are short claims and the documents are long abstracts, so a rotation
tuned on the document distribution may simply not fit the query distribution.
That failure would show up as itq/ste winning symmetric and losing asymmetric.

    python bench/learned_rotation_lfm25.py --emb-dir bench/.cache/LFM25_SCIFACT \
                                           --data-dir <scifact json dir>
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


# ---------------------------------------------------------------- transforms


def fit_haar(X, *, d, seed, **kw):
    return haar_rotation(d, seed=seed).astype(np.float32)


def fit_itq(X, *, d, seed, iters=30, **kw):
    """Gong & Lazebnik ITQ: alternate sign() with orthogonal Procrustes.

    Minimizes ||B - X R||_F over B in {-1,+1} and R orthogonal. Each iteration
    is one sign() and one SVD of a d x d matrix, so it is cheap and monotone --
    no learning rate, no early stopping, no way to overfit a hyperparameter.
    """
    R = haar_rotation(d, seed=seed).astype(np.float32)
    for _ in range(iters):
        B = np.where(X @ R > 0, 1.0, -1.0).astype(np.float32)
        U, _, Vt = np.linalg.svd(X.T @ B)
        R = (U @ Vt).astype(np.float32)
    return R


def fit_ste(X, *, d, seed, steps=400, batch=256, cands=64, margin=0.05,
            lr=1e-3, **kw):
    """Learn a linear map with sign() in the graph, via straight-through.

    Teacher is the fp32 similarity structure of the corpus itself: for each
    anchor, the nearest of a random candidate set is a positive and the
    furthest a negative, and the binary codes are pushed to preserve that
    ordering. No labels, no queries.

    The straight-through gradient is the hardtanh window (pass gradient where
    |x| <= 1, zero outside), which is the standard choice -- an unclipped
    identity STE lets already-saturated coordinates keep accumulating gradient
    and the map drifts without improving the sign pattern.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    class _Sign(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sign(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * (x.abs() <= 1.0).to(g.dtype)

    Xt = torch.from_numpy(X)
    n = Xt.shape[0]
    W = torch.nn.Parameter(torch.from_numpy(haar_rotation(d, seed=seed)).clone())
    opt = torch.optim.Adam([W], lr=lr)
    gen = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        a_idx = torch.randint(0, n, (batch,), generator=gen)
        c_idx = torch.randint(0, n, (batch, cands), generator=gen)
        A = Xt[a_idx]                                   # (B, d)
        C = Xt[c_idx]                                   # (B, cands, d)
        sims = torch.einsum("bd,bcd->bc", A, C)         # fp32 teacher
        pos = C[torch.arange(batch), sims.argmax(1)]
        neg = C[torch.arange(batch), sims.argmin(1)]

        bA, bP, bN = (_Sign.apply(t @ W) for t in (A, pos, neg))
        s_pos = (bA * bP).mean(1)
        s_neg = (bA * bN).mean(1)
        loss = torch.relu(margin - (s_pos - s_neg)).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return W.detach().numpy().astype(np.float32)


def fit_ste_orth(X, *, d, seed, steps=300, lr=1e-3, **kw):
    """STE with the map orthogonal BY CONSTRUCTION, via a Cayley transform.

    The free-W version (fit_ste) reaches a LOWER quantization error than ITQ and
    a catastrophically worse nDCG. Nothing stops an unconstrained linear map from
    collapsing the space: shrink the coordinates that are expensive to sign
    correctly and both the reconstruction term and the margin loss improve while
    the retrieval geometry is destroyed. Quantization error is a proxy, and an
    unconstrained optimizer wins the proxy.

    Projecting back to the orthogonal group every N steps does not work either --
    W becomes ill-conditioned fast enough that the SVD fails to converge before
    the first projection lands. So constrain it structurally instead:

        W = R0 @ (I - A)(I + A)^-1,   A = P - P^T  (skew-symmetric)

    The Cayley transform of any skew-symmetric matrix is orthogonal, so every
    point the optimizer can reach is a rotation. P is initialized at zero, so
    training starts exactly at the Haar baseline and can only move along the
    manifold -- which makes this a clean test of whether a *rotation* can be
    learned, separate from whether an arbitrary map can cheat.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    class _Sign(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sign(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * (x.abs() <= 1.0).to(g.dtype)

    Xt = torch.from_numpy(X)
    n = Xt.shape[0]
    R0 = torch.from_numpy(haar_rotation(d, seed=seed))
    I = torch.eye(d)
    P = torch.nn.Parameter(torch.zeros(d, d))
    opt = torch.optim.Adam([P], lr=lr)
    gen = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        A = P - P.T
        W = R0 @ torch.linalg.solve(I + A, I - A)
        a_idx = torch.randint(0, n, (256,), generator=gen)
        c_idx = torch.randint(0, n, (256, 64), generator=gen)
        Aa = Xt[a_idx]
        C = Xt[c_idx]
        sims = torch.einsum("bd,bcd->bc", Aa, C)
        pos = C[torch.arange(256), sims.argmax(1)]
        neg = C[torch.arange(256), sims.argmin(1)]
        bA, bP, bN = (_Sign.apply(t @ W) for t in (Aa, pos, neg))
        loss = torch.relu(0.05 - ((bA * bP).mean(1) - (bA * bN).mean(1))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        A = P - P.T
        W = R0 @ torch.linalg.solve(I + A, I - A)
    return W.numpy().astype(np.float32)


def orthogonality_error(R):
    """||R^T R - I||_F — 0 for a true rotation, large for a collapsed map."""
    d = R.shape[0]
    return float(np.linalg.norm(R.T @ R - np.eye(d, dtype=R.dtype)))


TRANSFORMS = {"haar": fit_haar, "itq": fit_itq, "ste": fit_ste,
              "ste-orth": fit_ste_orth}


# ---------------------------------------------------------------- evaluation


def score(R, Cc, Qc, corpus_ids, query_ids, rel, base):
    c_codes = encode_signs(Cc @ R)
    q_codes = encode_signs(Qc @ R)
    q_rot = Qc @ R
    sym = np.empty((len(Qc), len(Cc)), dtype=np.float64)
    asym = np.empty_like(sym)
    for i in range(len(Qc)):
        sym[i] = -hamming_distances(c_codes, q_codes[i])
        asym[i] = asymmetric_scores(q_rot[i], c_codes)
    return (evaluate(sym, corpus_ids, query_ids, rel, fp32_scores=base),
            evaluate(asym, corpus_ids, query_ids, rel, fp32_scores=base),
            c_codes.shape[1])


def quantization_error(R, Cc):
    """Mean ||sign(XR) - XR||^2 per row — what ITQ explicitly minimizes."""
    Z = Cc @ R
    return float(((np.sign(Z) - Z) ** 2).sum(axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="*", default=list(TRANSFORMS))
    ap.add_argument("--itq-iters", type=int, default=30)
    ap.add_argument("--ste-steps", type=int, default=400)
    ap.add_argument("--holdout", action="store_true",
                    help="fit on a random half of the documents")
    args = ap.parse_args()

    emb, data = pathlib.Path(args.emb_dir), pathlib.Path(args.data_dir)
    C = _unit(np.load(emb / "docs.npy").astype(np.float32))
    Q = _unit(np.load(emb / "queries.npy").astype(np.float32))
    corpus_ids = json.loads((emb / "doc_ids.json").read_text())
    query_ids = json.loads((emb / "query_ids.json").read_text())
    rel = json.loads((data / "qrels.json").read_text())

    base = fp32_scores(C, Q)
    fp32 = evaluate(base, corpus_ids, query_ids, rel, fp32_scores=base)
    mu = C.mean(axis=0)
    Cc, Qc = (C - mu).astype(np.float32), (Q - mu).astype(np.float32)
    d = C.shape[1]

    fit_on = Cc
    if args.holdout:
        rng = np.random.default_rng(args.seed)
        keep = rng.permutation(len(Cc))[: len(Cc) // 2]
        fit_on = Cc[keep]
        print(f"holdout: fitting on {len(fit_on)} of {len(Cc)} documents\n")

    print(f"fp32 baseline nDCG@10 {fp32['ndcg@10']:.4f}   "
          f"({d} dims, {d // 8} B/vec when binarized)\n")
    print(f"{'method':<9} {'fit s':>7} {'quant err':>10} {'orth err':>9} "
          f"{'symmetric':>10} {'asymmetric':>11} {'vs haar sym':>12} "
          f"{'vs haar asym':>13}")
    print("-" * 90)

    rows, ref = [], {}
    for name in args.methods:
        t0 = time.time()
        R = TRANSFORMS[name](fit_on, d=d, seed=args.seed,
                             iters=args.itq_iters, steps=args.ste_steps)
        fit_s = time.time() - t0
        ms, ma, bpv = score(R, Cc, Qc, corpus_ids, query_ids, rel, base)
        qe = quantization_error(R, Cc)
        if name == "haar":
            ref = {"sym": ms["ndcg@10"], "asym": ma["ndcg@10"]}
        ds = ms["ndcg@10"] - ref.get("sym", ms["ndcg@10"])
        da = ma["ndcg@10"] - ref.get("asym", ma["ndcg@10"])
        oe = orthogonality_error(R)
        print(f"{name:<9} {fit_s:>7.1f} {qe:>10.2f} {oe:>9.2f} "
              f"{ms['ndcg@10']:>10.4f} {ma['ndcg@10']:>11.4f} {ds:>+12.4f} "
              f"{da:>+13.4f}")
        rows.append({"method": name, "fit_seconds": round(fit_s, 2),
                     "bytes_per_vec": int(bpv), "quantization_error": qe,
                     "orthogonality_error": oe,
                     "symmetric": ms, "asymmetric": ma})

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_holdout" if args.holdout else ""
    (outdir / f"lfm25_learned_rotation{tag}.json").write_text(
        json.dumps({"fp32": fp32, "holdout": args.holdout, "rows": rows},
                   indent=2))
    print(f"\nwrote {outdir / f'lfm25_learned_rotation{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
