"""Does the SimHash collision guarantee survive a structured rotation? (remax#59)

remex#71 swapped Haar for a randomized Hadamard transform and measured a large
construction speedup with no recall cost. That argument does **not** transfer
here. remex needed an orthogonal map that makes coordinates ~N(0, 1/d) so its
Lloyd-Max codebook quantizes the distribution it was fitted to; reconstruction
MSE is indifferent to *which* isotropic rotation you use. remax rests on the
Charikar (2002) / Goemans-Williamson (1995) collision bound

    P[sign(<r, x>) != sign(<r, y>)] = theta / pi

which is a statement about the *distribution* of the projection directions. A
randomized Hadamard carries O(d log d) bits of randomness against Haar's
O(d^2), and its rows are structured rather than independent. So it has to be
measured.

Four steps, in the order that matters — theory first, recall last:

  1. ``collision``    Empirical collision rate vs the published 1 - theta/pi
                      curve, across a theta grid, at k in {1, 4, 8, 16}. The
                      anchor is the *published curve*, not Haar: two equally
                      structured rotations agreeing with each other would
                      prove nothing.
  2. ``spread``       The mean can match while the variance does not. Spread of
                      the collision estimator is what sets recall at fixed k.
  3. ``independence`` ``SeedSequence`` gives independent seeds; with a
                      structured transform the resulting hyperplane sets may
                      still be correlated. This is the assumption stacking
                      rests on, so it is checked directly.
  4. ``recall``       Only then, end-to-end recall@10 against exact-cosine
                      ground truth on real corpora, plus build cost.

Usage
-----
    python3 bench/rotation_lsh_fidelity.py                # steps 1-3 (synthetic)
    python3 bench/rotation_lsh_fidelity.py --steps recall # step 4, needs caches
    python3 bench/rotation_lsh_fidelity.py --steps all

Step 4 needs at least one embedding cache:

    bash bench/fetch_jina_v5_scifact_cache.sh   # d=768
    bash bench/fetch_gemini_cache.sh            # d=3072, truncated to 1024 too

Results: ``bench/results/ROTATION_LSH.md``.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from remax import StackedSignBitQuantizer
from remax.packing import hamming_distances, stable_top_k
from remax.rotation import _MIN_RHT_ROUNDS, _largest_pow2_divisor, rht_rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "bench" / ".cache"

THETA_FRACS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)
KINDS = ("haar", "rht")


# ------------------------------------------------------------------ #
# synthetic pair generation
# ------------------------------------------------------------------ #
def pairs_at_angle(theta, n_pairs, d, rng, kind="gauss"):
    """Unit vectors with exactly the requested pairwise angle.

    ``kind`` selects the input distribution:

    * ``gauss``  — isotropic, the textbook setting.
    * ``spiky``  — heavy-tailed coordinates (t-distributed), stressing the
      Berry-Esseen correction that a Rademacher-style direction relies on.
    * ``aniso``  — axis-aligned variance decay plus a shared mean direction.
      This is the regime real embeddings live in (the "rogue dimension"
      phenomenon: a handful of coordinates carrying outsized variance), and
      the one where a structured rotation is most exposed.
    """
    if kind == "gauss":
        U = rng.standard_normal((n_pairs, d))
        V = rng.standard_normal((n_pairs, d))
    elif kind == "spiky":
        U = rng.standard_normal((n_pairs, d)) / np.sqrt(
            rng.chisquare(3, size=(n_pairs, d)) / 3)
        V = rng.standard_normal((n_pairs, d)) / np.sqrt(
            rng.chisquare(3, size=(n_pairs, d)) / 3)
    elif kind == "aniso":
        scale = 1.0 / np.sqrt(np.arange(1, d + 1))
        mean_dir = rng.standard_normal(d)
        mean_dir /= np.linalg.norm(mean_dir)
        U = rng.standard_normal((n_pairs, d)) * scale + 3.0 * mean_dir
        V = rng.standard_normal((n_pairs, d)) * scale + 3.0 * mean_dir
    else:
        raise ValueError(f"unknown data kind {kind!r}")

    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V -= np.sum(V * U, axis=1, keepdims=True) * U
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Y = np.cos(theta) * U + np.sin(theta) * V
    return U.astype(np.float32), Y.astype(np.float32)


def collision_fracs(X, Y, rot_matrix, d, k):
    """Per-stack disagreement fractions, shape (n_pairs, k)."""
    sx = (X @ rot_matrix) > 0
    sy = (Y @ rot_matrix) > 0
    return (sx != sy).reshape(X.shape[0], k, d).mean(axis=2)


def rot_matrix_for(kind, d, k, seed, rounds=None):
    """(d, k*d) projection matrix; honours an explicit RHT round count."""
    if rounds is None:
        return StackedSignBitQuantizer(
            d=d, k=k, seed=seed, rotation=kind)._rotation_matrix
    # Explicit rounds are only reachable for the diagnostic sweep below; the
    # library floors them at _MIN_RHT_ROUNDS on purpose.
    states = np.random.SeedSequence(seed).generate_state(k, dtype=np.uint32)
    out = np.empty((d, k * d), np.float32)
    for j in range(k):
        out[:, j * d:(j + 1) * d] = _unguarded_rht(d, int(states[j]), rounds)
    return out


def _unguarded_rht(d, seed, rounds):
    """RHT with an arbitrary round count, for the rounds<2 diagnostic only."""
    if rounds >= _MIN_RHT_ROUNDS:
        return rht_rotation(d, seed=seed, rounds=rounds)
    from remax.rotation import _fwht_inplace
    B = _largest_pow2_divisor(d)
    rng = np.random.default_rng(seed)
    Y = np.eye(d, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(B))
    for _ in range(rounds):
        Y = Y[:, rng.permutation(d)] * rng.choice(
            np.array([-1.0, 1.0], np.float32), size=d)
        Y = np.ascontiguousarray(Y.reshape(d, d // B, B))
        _fwht_inplace(Y)
        Y = Y.reshape(d, d) * scale
    return Y


# ------------------------------------------------------------------ #
# step 1: collision rate vs the published curve
# ------------------------------------------------------------------ #
def step_collision(d, ks, n_pairs, n_seeds, data_kinds):
    print(f"\n{'='*88}\nSTEP 1 — collision rate vs theta/pi   "
          f"d={d}  pairs={n_pairs}  rotation seeds={n_seeds}\n{'='*88}")
    rng = np.random.default_rng(0xC0FFEE)
    kmax = max(ks)
    mats = {kind: [rot_matrix_for(kind, d, kmax, 1000 + s) for s in range(n_seeds)]
            for kind in KINDS}
    for data_kind in data_kinds:
        print(f"\n--- input: {data_kind} ---")
        print(f"{'theta/pi':>9} {'rot':>5} " +
              " ".join(f"{'k='+str(k):>18}" for k in ks))
        for frac in THETA_FRACS:
            X, Y = pairs_at_angle(math.pi * frac, n_pairs, d, rng, data_kind)
            for kind in KINDS:
                cells = []
                for k in ks:
                    est = np.concatenate([
                        collision_fracs(X, Y, mats[kind][s][:, :k * d], d, k
                                        ).mean(axis=1)
                        for s in range(n_seeds)])
                    cells.append(f"{est.mean():.4f}+-{est.std():.4f}"
                                 f"{est.mean()-frac:+.4f}")
                print(f"{frac:9.4f} {kind:>5} " +
                      " ".join(f"{c:>18}" for c in cells))


def step_bias_vs_k(d, ks, n_pairs, n_seeds, data_kind, round_opts):
    """Does the bias shrink as k grows?

    Sampling error must. A defect shared by every rotation in the stack
    cannot — averaging k copies of the same structural bias leaves it
    untouched. That distinction is the sharpest single signal separating a
    usable structured rotation from an unusable one, so it gets its own
    sweep with the round count varied explicitly.
    """
    print(f"\n{'='*88}\nSTEP 1b — does the bias shrink with k?   "
          f"d={d} input={data_kind}\n"
          f"  sampling error shrinks; a shared structural defect does not\n"
          f"{'='*88}")
    rng = np.random.default_rng(0xA11CE)
    kmax = max(ks)
    variants = [("haar", None)] + [(f"rht-r{r}", r) for r in round_opts]
    mats = {name: [rot_matrix_for("haar" if name == "haar" else "rht",
                                  d, kmax, 4000 + s, rounds=r)
                   for s in range(n_seeds)]
            for name, r in variants}
    for frac in (0.35, 0.5):
        print(f"\n  theta/pi = {frac}")
        print(f"{'rot':>9} " + " ".join(f"{'k='+str(k):>12}" for k in ks))
        X, Y = pairs_at_angle(math.pi * frac, n_pairs, d, rng, data_kind)
        for name, _ in variants:
            cells = []
            for k in ks:
                est = np.concatenate([
                    collision_fracs(X, Y, m[:, :k * d], d, k).mean(axis=1)
                    for m in mats[name]])
                cells.append(f"{est.mean() - frac:+.4f}")
            print(f"{name:>9} " + " ".join(f"{c:>12}" for c in cells))


# ------------------------------------------------------------------ #
# steps 2 + 3: spread and cross-stack independence
# ------------------------------------------------------------------ #
def step_spread_independence(d, k, n_pairs, n_seeds, data_kind, round_opts):
    print(f"\n{'='*94}\nSTEPS 2+3 — estimator spread and stack independence   "
          f"d={d} k={k} input={data_kind}\n"
          f"  sd_ratio  = observed spread / binomial sqrt(p(1-p)/(k*d)); "
          f"<1 means orthogonal directions beat independent ones\n"
          f"  var_ratio = Var(pooled) / (Var(one stack) / k); "
          f"1.0 == the k stacks are independent\n{'='*94}")
    rng = np.random.default_rng(0xBEEF)
    variants = [("haar", None)] + [(f"rht-r{r}", r) for r in round_opts]
    mats = {name: [rot_matrix_for("haar" if name == "haar" else "rht",
                                  d, k, 2000 + s, rounds=r)
                   for s in range(n_seeds)]
            for name, r in variants}
    print(f"{'theta/pi':>9} {'rot':>9} {'bias':>9} {'sd_obs':>9} {'sd_binom':>9} "
          f"{'sd_ratio':>9} {'mean|corr|':>11} {'var_ratio':>10}")
    for frac in THETA_FRACS:
        p = frac
        sd_binom = math.sqrt(p * (1 - p) / (k * d))
        X, Y = pairs_at_angle(math.pi * frac, n_pairs, d, rng, data_kind)
        for name, _ in variants:
            per_seed = [collision_fracs(X, Y, m, d, k) for m in mats[name]]
            pooled = np.concatenate([f.mean(axis=1) for f in per_seed])
            corrs, vrs = [], []
            for f in per_seed:
                c = np.corrcoef(f.T)
                corrs.append(np.abs(c[~np.eye(k, dtype=bool)]).mean())
                vrs.append(f.mean(axis=1).var() / (f.var(axis=0).mean() / k))
            print(f"{frac:9.4f} {name:>9} {pooled.mean()-p:+9.4f} "
                  f"{pooled.std():9.5f} {sd_binom:9.5f} "
                  f"{pooled.std()/sd_binom:9.3f} {np.mean(corrs):11.4f} "
                  f"{np.mean(vrs):10.3f}")
        print()


# ------------------------------------------------------------------ #
# step 4: end-to-end recall on real embeddings
# ------------------------------------------------------------------ #
def _l2norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_corpora(n_queries=300):
    """Available real-embedding corpora as {name: (corpus, queries)}."""
    out = {}
    jina = CACHE / "JINA_V5_BEIR_SCIFACT"
    if (jina / "corpus.npy").exists():
        out["jina-v5-nano/scifact d=768"] = (
            _l2norm(np.load(jina / "corpus.npy").astype(np.float32)),
            _l2norm(np.load(jina / "queries.npy").astype(np.float32)))
    gem = CACHE / "GEMINI" / "embeddings.npy"
    if gem.exists():
        G = np.load(gem).astype(np.float32)
        rng = np.random.default_rng(7)
        qi = rng.choice(G.shape[0], n_queries, replace=False)
        mask = np.ones(G.shape[0], bool)
        mask[qi] = False
        for dd in (1024, 3072):
            # Gemini embeddings are Matryoshka: truncate, then renormalize.
            out[f"gemini d={dd}"] = (_l2norm(G[mask][:, :dd].copy()),
                                     _l2norm(G[qi][:, :dd].copy()))
    return out


def exact_topn(C, Q, n):
    sims = Q @ C.T
    idx = np.argpartition(-sims, n, axis=1)[:, :n]
    order = np.take_along_axis(sims, idx, 1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


def _encode(X, q, batch=2048):
    out = np.empty((X.shape[0], q.n_bits // 8), np.uint8)
    for s in range(0, X.shape[0], batch):
        out[s:s + batch] = q.encode(X[s:s + batch])
    return out


def recall_at_n(C, Q, gt, q, n):
    cc, qc = _encode(C, q), _encode(Q, q)
    hits = sum(len(np.intersect1d(stable_top_k(hamming_distances(cc, qc[i]), n),
                                  gt[i]))
               for i in range(Q.shape[0]))
    return hits / (Q.shape[0] * n)


def step_recall(ks, seeds, topn):
    corpora = load_corpora()
    if not corpora:
        print("\nSTEP 4 skipped: no embedding cache found under bench/.cache/.\n"
              "  bash bench/fetch_jina_v5_scifact_cache.sh\n"
              "  bash bench/fetch_gemini_cache.sh")
        return
    for name, (C, Q) in corpora.items():
        d = C.shape[1]
        var = C.var(axis=0)
        print(f"\n{'='*88}\nSTEP 4 — recall@{topn} vs exact cosine   {name}\n"
              f"  corpus {C.shape}, queries {Q.shape}, FWHT block B="
              f"{_largest_pow2_divisor(d)}\n"
              f"  anisotropy: max coord variance / mean = "
              f"{var.max()/var.mean():.1f}x\n{'='*88}")
        gt = exact_topn(C, Q, topn)
        results, build = {}, {}
        for kind in KINDS:
            per_k = {}
            ts = []
            for s in seeds:
                t0 = time.perf_counter()
                qs = {k: StackedSignBitQuantizer(d=d, k=k, seed=s, rotation=kind)
                      for k in ks}
                ts.append(time.perf_counter() - t0)
                for k in ks:
                    per_k.setdefault(k, []).append(
                        recall_at_n(C, Q, gt, qs[k], topn))
                del qs
            results[kind] = per_k
            build[kind] = np.mean(ts)
        print(f"{'rotation':>9} {'build':>9} " +
              " ".join(f"{'k='+str(k):>16}" for k in ks))
        for kind in KINDS:
            print(f"{kind:>9} {build[kind]:8.2f}s " + " ".join(
                f"{np.mean(results[kind][k]):.4f}+-{np.std(results[kind][k]):.4f}"
                for k in ks))
        print(f"{'delta':>9} {build['haar']/build['rht']:7.2f}x " + " ".join(
            f"{np.mean(results['rht'][k]) - np.mean(results['haar'][k]):+16.4f}"
            for k in ks))


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default="theory",
                    choices=["theory", "recall", "all"],
                    help="'theory' = steps 1-3 (synthetic, no cache needed)")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    if args.steps in ("theory", "all"):
        for d in (768, 1024):
            step_collision(d, [1, 4, 8, 16], args.pairs, args.seeds,
                           ["gauss", "aniso", "spiky"])
        # rounds=1 is what remex#71 uses when d is a power of two. d=1024 is
        # exactly that case, and it is a mainstream embedding width.
        step_bias_vs_k(1024, [1, 4, 8, 16], args.pairs, args.seeds, "aniso",
                       [1, 2])
        step_spread_independence(1024, 8, args.pairs, args.seeds, "aniso",
                                 [1, 2, 3])
        step_spread_independence(768, 8, args.pairs, args.seeds, "aniso", [2, 3])
        step_spread_independence(1024, 8, args.pairs, args.seeds, "gauss", [1, 2])
    if args.steps in ("recall", "all"):
        step_recall([1, 2, 4, 8], list(range(11, 11 + args.seeds)), args.topn)
    print(f"\nelapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
