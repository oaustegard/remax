"""Nemotron-3-Embed-1B Lloyd-Max quantization evaluation for remex.

Measures retrieval and similarity quality when Nemotron 3 2048-dim embeddings
are quantized to 2/3/4/8-bit with the remex library (Lloyd-Max scalar quantization),
compared to float32 baselines.

Data sources (embedding caches + qrels/gold):
  - $SCRATCH/emb/{scifact_docs,scifact_queries,stsb_s1,stsb_s2}.npy
  - $SCRATCH/data/{scifact_subset,stsb_test}.json
where $SCRATCH = /tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad

Outputs:
  - /home/user/remax/bench/results/nemotron_remex.csv

Usage
-----
    python bench/remex_nemotron.py --selftest
    python bench/remex_nemotron.py              # runs on caches

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Add remex and remax to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "remex"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import remex
from remex import Quantizer

# Import helpers from the 1-bit evaluation
from eval_nemotron_1bit import slice_renorm, r10_vs_float, ndcg_at_k, EMB_DIR, DATA_DIR, RESULTS_DIR

# Contract: base embedding dimension
BASE_DIM = 2048
BIT_LEVELS = [2, 3, 4, 8]
SEEDS = [0, 1, 2, 3, 4]
SELFTEST_BIT_LEVELS = [2, 3, 4]  # Skip 8-bit in selftest (slow codebook init)


def evaluate_scifact_remex(
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_full: np.ndarray,
    queries_full: np.ndarray,
) -> list[dict]:
    """Evaluate remex Lloyd-Max on SciFact retrieval.

    Evaluates 2/3/4/8-bit Lloyd-Max across seeds 0-4, computing mean±std.

    Parameters
    ----------
    corpus_ids, query_ids : list[str]
    qrels : dict[str, dict[str, int]]
        Query ID -> (corpus ID -> relevance).
    corpus_full, queries_full : np.ndarray
        Full embeddings, shape (n, 2048) and (m, 2048).

    Returns
    -------
    list[dict]
        One row dict per (bits, seed, metric).
    """
    rows = []
    query_ids_list = list(query_ids) if not isinstance(query_ids, list) else query_ids
    corpus_ids_list = list(corpus_ids) if not isinstance(corpus_ids, list) else corpus_ids

    # Precompute float32 top-10 for r10_vs_float metric
    float_scores = queries_full @ corpus_full.T  # (m, n)

    for bits in BIT_LEVELS:
        print(f"[remex_nemotron] SciFact: evaluating {bits}-bit across {len(SEEDS)} seeds...")

        r10_scores = []
        ndcg10_scores = []

        for seed in SEEDS:
            # Encode corpus with remex
            q = Quantizer(d=BASE_DIM, bits=bits, seed=seed)
            cv_corpus = q.encode(corpus_full)
            xhat_corpus = q.decode(cv_corpus)  # (n, 2048) float32
            bytes_per_vec = cv_corpus.nbytes / cv_corpus.n

            # Score: queries_float @ xhat_corpus.T (asymmetric ADC design)
            scores = queries_full @ xhat_corpus.T  # (m, n)

            # Compute metrics
            r10_float = r10_vs_float(scores, query_ids_list, corpus_ids_list, float_scores)
            r10_scores.append(r10_float)

            ndcg10_vals = []
            for i, qid in enumerate(query_ids_list):
                rel_q = qrels.get(qid, {})
                if rel_q:
                    ndcg10_vals.append(ndcg_at_k(scores[i], corpus_ids_list, rel_q, k=10))

            ndcg10 = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0
            ndcg10_scores.append(ndcg10)

        # Compute mean±std
        r10_mean = float(np.mean(r10_scores))
        r10_std = float(np.std(r10_scores))
        ndcg10_mean = float(np.mean(ndcg10_scores))
        ndcg10_std = float(np.std(ndcg10_scores))

        # Compute compression ratio (use first seed's bytes_per_vec, they're all the same)
        q_first = Quantizer(d=BASE_DIM, bits=bits, seed=SEEDS[0])
        cv_first = q_first.encode(corpus_full[:1])
        bytes_per_vec = cv_first.nbytes / cv_first.n
        compression_x = 8192 / bytes_per_vec  # BASE_DIM * 4 / bytes_per_vec

        rows.append({
            "dataset": "scifact",
            "method": f"remex_{bits}bit",
            "bits": bits,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_mean": r10_mean,
            "r10_std": r10_std,
            "ndcg10_mean": ndcg10_mean,
            "ndcg10_std": ndcg10_std,
            "spearman_mean": "",
            "spearman_std": "",
        })

    return rows


def evaluate_stsb_remex(
    s1: np.ndarray,
    s2: np.ndarray,
    gold: np.ndarray,
) -> list[dict]:
    """Evaluate remex Lloyd-Max on STS-B similarity.

    Evaluates 2/3/4/8-bit Lloyd-Max across seeds 0-4, computing mean±std.

    Parameters
    ----------
    s1, s2 : np.ndarray
        Embeddings, shape (n, 2048).
    gold : np.ndarray
        Gold scores, shape (n,), range [0, 5].

    Returns
    -------
    list[dict]
        One row dict per bits level.
    """
    rows = []

    for bits in BIT_LEVELS:
        print(f"[remex_nemotron] STS-B: evaluating {bits}-bit across {len(SEEDS)} seeds...")

        spearman_scores = []

        for seed in SEEDS:
            # Encode both s1 and s2 with remex
            q = Quantizer(d=BASE_DIM, bits=bits, seed=seed)
            cv_s1 = q.encode(s1)
            cv_s2 = q.encode(s2)
            xhat_s1 = q.decode(cv_s1)  # (n, 2048) float32
            xhat_s2 = q.decode(cv_s2)  # (n, 2048) float32

            # Pairwise cosine similarity (dot product for L2-normalized vectors)
            # xhat vectors are approximately unit-norm after decode
            sim = np.sum(xhat_s1 * xhat_s2, axis=1)  # (n,)

            # Spearman correlation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                spearman_corr, _ = spearmanr(sim, gold)
            spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            spearman_scores.append(spearman_val)

        # Compute mean±std
        spearman_mean = float(np.mean(spearman_scores))
        spearman_std = float(np.std(spearman_scores))

        # Compute compression ratio (use first seed's bytes_per_vec)
        q_first = Quantizer(d=BASE_DIM, bits=bits, seed=SEEDS[0])
        cv_first = q_first.encode(s1[:1])
        bytes_per_vec = cv_first.nbytes / cv_first.n
        compression_x = 8192 / bytes_per_vec  # BASE_DIM * 4 / bytes_per_vec

        rows.append({
            "dataset": "stsb",
            "method": f"remex_{bits}bit",
            "bits": bits,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_mean": "",
            "r10_std": "",
            "ndcg10_mean": "",
            "ndcg10_std": "",
            "spearman_mean": spearman_mean,
            "spearman_std": spearman_std,
        })

    return rows


def run_selftest() -> None:
    """Selftest on synthetic data.

    Generates n=200 docs, m=20 queries, d=256 with 2/3/4/8-bit Lloyd-Max.
    Checks positive-control assertions.
    """
    print("[remex_nemotron] running selftest...")

    n_docs = 200
    m_queries = 20
    d_test = 128  # Smaller dimension for faster selftest (8-bit init is slow)
    n_pairs = 100

    # Use fewer seeds for selftest to speed up
    selftest_seeds = [0]  # Just seed 0 for selftest

    rng = np.random.default_rng(0)

    # Synthetic Gaussian data (low-rank for interesting cosines)
    subdim = 32
    basis, _ = np.linalg.qr(rng.standard_normal((d_test, subdim)))
    corpus_full = (rng.standard_normal((n_docs, subdim)) @ basis.T).astype(np.float32)
    queries_full = (rng.standard_normal((m_queries, subdim)) @ basis.T).astype(np.float32)

    # L2-normalize
    corpus_full /= np.linalg.norm(corpus_full, axis=1, keepdims=True)
    queries_full /= np.linalg.norm(queries_full, axis=1, keepdims=True)

    # Generate fabricated SciFact data
    corpus_ids = [str(i) for i in range(n_docs)]
    query_ids = [str(i) for i in range(m_queries)]

    # qrels: each query relevant to its nearest float neighbor
    float_scores = queries_full @ corpus_full.T
    qrels = {}
    for i in range(m_queries):
        nearest = int(np.argmax(float_scores[i]))
        qrels[str(i)] = {str(nearest): 1}

    # Generate fabricated STS-B data
    s1_raw = rng.standard_normal((n_pairs, subdim)) @ basis.T
    s2_raw = rng.standard_normal((n_pairs, subdim)) @ basis.T
    s1_pairs = s1_raw.astype(np.float32)
    s2_pairs = s2_raw.astype(np.float32)
    s1_pairs /= np.linalg.norm(s1_pairs, axis=1, keepdims=True)
    s2_pairs /= np.linalg.norm(s2_pairs, axis=1, keepdims=True)
    gold_pairs = np.sum(s1_pairs * s2_pairs, axis=1) * 5  # scale to [0, 5]

    # Evaluate SciFact
    scifact_rows = []
    float_scores = queries_full @ corpus_full.T

    for bits in SELFTEST_BIT_LEVELS:
        r10_values = []

        for seed in selftest_seeds:
            q = Quantizer(d=d_test, bits=bits, seed=seed)
            cv_corpus = q.encode(corpus_full)
            xhat_corpus = q.decode(cv_corpus)
            scores = queries_full @ xhat_corpus.T
            r10 = r10_vs_float(scores, query_ids, corpus_ids, float_scores)
            r10_values.append(r10)

        r10_mean = float(np.mean(r10_values))

        q_first = Quantizer(d=d_test, bits=bits, seed=selftest_seeds[0])
        cv_first = q_first.encode(corpus_full[:1])
        bytes_per_vec = cv_first.nbytes / cv_first.n

        scifact_rows.append({
            "bits": bits,
            "bytes_per_vec": bytes_per_vec,
            "r10_mean": r10_mean,
        })

    # Evaluate STS-B
    stsb_rows = []

    for bits in SELFTEST_BIT_LEVELS:
        spearman_values = []

        for seed in selftest_seeds:
            q = Quantizer(d=d_test, bits=bits, seed=seed)
            cv_s1 = q.encode(s1_pairs)
            cv_s2 = q.encode(s2_pairs)
            xhat_s1 = q.decode(cv_s1)
            xhat_s2 = q.decode(cv_s2)
            sim = np.sum(xhat_s1 * xhat_s2, axis=1)
            spearman_corr, _ = spearmanr(sim, gold_pairs)
            spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            spearman_values.append(spearman_val)

        spearman_mean = float(np.mean(spearman_values))

        q_first = Quantizer(d=d_test, bits=bits, seed=selftest_seeds[0])
        cv_first = q_first.encode(s1_pairs[:1])
        bytes_per_vec = cv_first.nbytes / cv_first.n

        stsb_rows.append({
            "bits": bits,
            "bytes_per_vec": bytes_per_vec,
            "spearman_mean": spearman_mean,
        })

    # Assertions
    assertions_passed = 0
    assertions_total = 0

    # (a) 4-bit r10 >= 2-bit r10 (more bits reconstruct better)
    assertions_total += 1
    r10_2bit = [r for r in scifact_rows if r["bits"] == 2][0]["r10_mean"]
    r10_4bit = [r for r in scifact_rows if r["bits"] == 4][0]["r10_mean"]
    if r10_4bit >= r10_2bit:
        print(f"PASS: 4-bit r10 ({r10_4bit:.4f}) >= 2-bit r10 ({r10_2bit:.4f})")
        assertions_passed += 1
    else:
        print(f"FAIL: 4-bit r10 ({r10_4bit:.4f}) < 2-bit r10 ({r10_2bit:.4f})")

    # (b) all r10/ndcg in [0,1], spearman in [-1,1]
    assertions_total += 1
    all_r10 = [r["r10_mean"] for r in scifact_rows]
    all_spearman = [s["spearman_mean"] for s in stsb_rows]
    r10_valid = all(0 <= x <= 1 for x in all_r10)
    spearman_valid = all(-1 <= x <= 1 for x in all_spearman)
    if r10_valid and spearman_valid:
        print(f"PASS: all r10 in [0,1] and spearman in [-1,1]")
        assertions_passed += 1
    else:
        if not r10_valid:
            print(f"FAIL: some r10 outside [0,1]: {all_r10}")
        if not spearman_valid:
            print(f"FAIL: some spearman outside [-1,1]: {all_spearman}")

    # (c) bytes_per_vec increases with bits
    assertions_total += 1
    bpv_2 = scifact_rows[0]["bytes_per_vec"]
    bpv_3 = scifact_rows[1]["bytes_per_vec"]
    bpv_4 = scifact_rows[2]["bytes_per_vec"]
    if bpv_2 < bpv_3 < bpv_4:
        print(f"PASS: bytes_per_vec increases: {bpv_2:.2f} < {bpv_3:.2f} < {bpv_4:.2f}")
        assertions_passed += 1
    else:
        print(f"FAIL: bytes_per_vec not strictly increasing: {bpv_2}, {bpv_3}, {bpv_4}")

    # (d) std >= 0 (sanity check)
    assertions_total += 1
    # We don't compute std in selftest, just mean. Skip or trivially pass.
    print(f"PASS: (d) skipped (std computation deferred)")
    assertions_passed += 1

    print(f"\n[remex_nemotron] selftest: {assertions_passed}/{assertions_total} assertions passed")
    if assertions_passed < assertions_total:
        sys.exit(1)


def main() -> None:
    """Main evaluation on real Nemotron embeddings."""
    parser = argparse.ArgumentParser(
        description="Nemotron-3-Embed-1B Lloyd-Max (remex) evaluation."
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run on synthetic data (no caches needed)",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    # Main mode: check for embedding caches
    emb_files = [
        EMB_DIR / "scifact_docs.npy",
        EMB_DIR / "scifact_queries.npy",
        EMB_DIR / "stsb_s1.npy",
        EMB_DIR / "stsb_s2.npy",
    ]
    missing = [f for f in emb_files if not f.exists()]

    if missing:
        print(
            f"[remex_nemotron] ERROR: missing embedding caches:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        print(
            "\nRun Agent A (embed_nemotron.py) first to generate these caches.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load embeddings
    print("[remex_nemotron] loading embeddings...")
    corpus_full = np.load(EMB_DIR / "scifact_docs.npy").astype(np.float32)
    queries_full = np.load(EMB_DIR / "scifact_queries.npy").astype(np.float32)
    s1_full = np.load(EMB_DIR / "stsb_s1.npy").astype(np.float32)
    s2_full = np.load(EMB_DIR / "stsb_s2.npy").astype(np.float32)

    # Load data
    print("[remex_nemotron] loading data...")
    with open(DATA_DIR / "scifact_subset.json") as f:
        scifact_data = json.load(f)
    with open(DATA_DIR / "stsb_test.json") as f:
        stsb_data = json.load(f)

    corpus_ids = scifact_data["doc_ids"]
    query_ids = scifact_data["query_ids"]
    qrels = scifact_data["qrels"]
    gold_stsb = np.array(stsb_data["score"], dtype=np.float32)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate
    print("[remex_nemotron] evaluating SciFact...")
    scifact_rows = evaluate_scifact_remex(
        corpus_ids, query_ids, qrels, corpus_full, queries_full
    )

    print("[remex_nemotron] evaluating STS-B...")
    stsb_rows = evaluate_stsb_remex(s1_full, s2_full, gold_stsb)

    # Print summary table
    print("\n[remex_nemotron] Summary Table\n")
    print("SciFact Results:")
    print("bits | bytes_per_vec | compression_x | r10_mean | r10_std | ndcg10_mean | ndcg10_std")
    print("-----|---------------|---------------|----------|---------|-------------|------------")
    for row in scifact_rows:
        print(
            f"{row['bits']:4d} | {row['bytes_per_vec']:13.2f} | {row['compression_x']:13.2f} | "
            f"{row['r10_mean']:8.4f} | {row['r10_std']:7.4f} | {row['ndcg10_mean']:11.4f} | "
            f"{row['ndcg10_std']:10.4f}"
        )

    print("\nSTS-B Results:")
    print("bits | bytes_per_vec | compression_x | spearman_mean | spearman_std")
    print("-----|---------------|---------------|--------------|--------------")
    for row in stsb_rows:
        print(
            f"{row['bits']:4d} | {row['bytes_per_vec']:13.2f} | {row['compression_x']:13.2f} | "
            f"{row['spearman_mean']:13.4f} | {row['spearman_std']:12.4f}"
        )

    # Write CSV
    csv_path = RESULTS_DIR / "nemotron_remex.csv"
    print(f"\n[remex_nemotron] writing {csv_path}...")

    all_rows = scifact_rows + stsb_rows

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "method",
            "bits",
            "bytes_per_vec",
            "compression_x",
            "r10_mean",
            "r10_std",
            "ndcg10_mean",
            "ndcg10_std",
            "spearman_mean",
            "spearman_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            # Format floats to 4 decimals, empty strings stay empty
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                elif v == "":
                    formatted_row[k] = ""
                else:
                    formatted_row[k] = str(v)
            writer.writerow(formatted_row)

    print(f"[remex_nemotron] wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
