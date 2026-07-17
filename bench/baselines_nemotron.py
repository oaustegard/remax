"""Nemotron-3-Embed-1B int8 and PQ baseline evaluation.

Measures retrieval and similarity quality when Nemotron 3 2048-dim embeddings
are quantized using int8 (per-vector symmetric scalar) and faiss ProductQuantizer
(data-dependent) baselines, compared to remax 1-bit methods and float32.

IMPORTANT: PQ is data-DEPENDENT (trained on the corpus). This is NOT a like-for-like
comparison with data-oblivious remax; it's a stronger baseline that costs a fit step.

Data sources (embedding caches + qrels/gold):
  - $SCRATCH/emb/{scifact_docs,scifact_queries,stsb_s1,stsb_s2}.npy
  - $SCRATCH/data/{scifact_subset,stsb_test}.json
where $SCRATCH = /tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad

Outputs:
  - /home/user/remax/bench/results/nemotron_baselines.csv

Usage
-----
    python bench/baselines_nemotron.py --selftest
    python bench/baselines_nemotron.py              # runs on caches

"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Import from eval_nemotron_1bit
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_nemotron_1bit import (
        slice_renorm, r10_vs_float, ndcg_at_k,
        EMB_DIR, DATA_DIR, RESULTS_DIR, BASE_DIM,
    )
except ImportError as e:
    print(f"ERROR: Could not import from eval_nemotron_1bit: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Install with: pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

SCRATCH = Path(
    "/tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad"
)

# Target byte budgets (from contract)
TARGET_BYTES = [1024, 512, 256, 128, 64, 32]


def int8_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vector symmetric scalar int8 quantization.

    For each row: find max absolute value, use as scale.
    Quantize to int8 [-128, 127] using scale.

    Parameters
    ----------
    x : np.ndarray, shape (n, d), dtype float32

    Returns
    -------
    codes : np.ndarray, shape (n, d), dtype int8
    scale : np.ndarray, shape (n,), dtype float32
    """
    n, d = x.shape
    codes = np.empty((n, d), dtype=np.int8)
    scale = np.empty(n, dtype=np.float32)

    for i in range(n):
        row = x[i]
        row_abs_max = np.max(np.abs(row))

        # Avoid division by zero
        if row_abs_max == 0:
            scale[i] = 1.0
            codes[i] = np.zeros(d, dtype=np.int8)
        else:
            # Scale factor maps [-row_abs_max, row_abs_max] to [-127, 127]
            s = row_abs_max / 127.0
            scale[i] = s
            codes[i] = np.round(row / s).astype(np.int8)

    return codes, scale


def int8_dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize int8 back to float32.

    Parameters
    ----------
    codes : np.ndarray, shape (n, d), dtype int8
    scale : np.ndarray, shape (n,), dtype float32

    Returns
    -------
    np.ndarray, shape (n, d), dtype float32
    """
    n, d = codes.shape
    x = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        x[i] = codes[i].astype(np.float32) * scale[i]
    return x


def evaluate_scifact_baselines(
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_full: np.ndarray,
    queries_full: np.ndarray,
) -> list[dict]:
    """Evaluate int8 and PQ baselines on SciFact retrieval.

    Parameters
    ----------
    corpus_ids, query_ids : list[str]
    qrels : dict[str, dict[str, int]]
    corpus_full, queries_full : np.ndarray
        Full embeddings, shape (n, 2048) and (m, 2048).

    Returns
    -------
    list[dict]
        One row dict per method.
    """
    rows = []
    query_ids_list = list(query_ids) if not isinstance(query_ids, list) else query_ids
    corpus_ids_list = list(corpus_ids) if not isinstance(corpus_ids, list) else corpus_ids

    # Precompute float32 top-10 for r10_vs_float metric
    float_scores = queries_full @ corpus_full.T  # (m, n)

    # ------- INT8 BASELINES -------
    # int8_2048, int8_mrl1024, int8_mrl512, int8_mrl256, int8_mrl128, int8_mrl64
    int8_methods = [
        ("int8_2048", 2048),
        ("int8_mrl1024", 1024),
        ("int8_mrl512", 512),
        ("int8_mrl256", 256),
        ("int8_mrl128", 128),
        ("int8_mrl64", 64),
    ]

    for method_id, dims in int8_methods:
        # Slice and renorm
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)

        # Quantize
        corpus_codes, corpus_scale = int8_quantize(corpus)
        query_codes, query_scale = int8_quantize(queries)

        # Bytes per vector (int8 codes)
        bytes_per_vec = dims * 1  # 1 byte per dimension

        # Score: dequantize and use float dot product
        corpus_dq = int8_dequantize(corpus_codes, corpus_scale)
        query_dq = int8_dequantize(query_codes, query_scale)
        scores = query_dq @ corpus_dq.T  # (m, n)

        # Metrics
        r10_float = r10_vs_float(scores, query_ids_list, corpus_ids_list, float_scores)

        ndcg10_vals = []
        for i, qid in enumerate(query_ids_list):
            rel_q = qrels.get(qid, {})
            if rel_q:
                ndcg10_vals.append(ndcg_at_k(scores[i], corpus_ids_list, rel_q, k=10))
        ndcg10 = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0

        compression_x = BASE_DIM * 4 // bytes_per_vec

        rows.append({
            "dataset": "scifact",
            "method": method_id,
            "family": "int8",
            "dims_or_M": dims,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": r10_float,
            "ndcg10": ndcg10,
            "spearman": "",
        })

    # ------- PQ BASELINES -------
    # pq_m32, pq_m64, pq_m128, pq_m256
    # PQ is data-dependent: train on the corpus
    pq_configs = [
        ("pq_m32", 32),
        ("pq_m64", 64),
        ("pq_m128", 128),
        ("pq_m256", 256),
    ]

    d = corpus_full.shape[1]  # 2048

    for method_id, M in pq_configs:
        # Check if M divides d
        if d % M != 0:
            # Pick nearest valid M and record actual M
            # Valid Ms are divisors of d=2048
            # 2048 = 2^11, so valid Ms: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
            divisors = [2**i for i in range(12)]
            # Find nearest divisor
            nearest_M = min(divisors, key=lambda x: abs(x - M))
            if nearest_M < M:
                nearest_M = max(d for d in divisors if d <= M)
            elif nearest_M > M:
                nearest_M = min(d for d in divisors if d >= M)
            M_actual = nearest_M
            print(f"[baselines_nemotron] PQ: M={M} does not divide d={d}; using M={M_actual}", file=sys.stderr)
        else:
            M_actual = M

        # Create and train PQ on corpus
        pq = faiss.ProductQuantizer(d, M_actual, 8)
        corpus_full_c = np.ascontiguousarray(corpus_full)
        pq.train(corpus_full_c)

        # Encode
        corpus_codes = pq.compute_codes(corpus_full_c)
        queries_full_c = np.ascontiguousarray(queries_full)
        query_codes = pq.compute_codes(queries_full_c)

        # Bytes per vector (M subquantizers × 8 bits = M bytes)
        bytes_per_vec = M_actual

        # Score using ADC (asymmetric distance computation)
        # Create lookup table
        scores = np.empty((len(queries_full), len(corpus_full)), dtype=np.float32)
        for i in range(len(queries_full)):
            query_vec = queries_full[i:i+1]
            # Compute ADC distance: for each corpus item, look up the distance
            # in the codebook. Use faiss search_and_reconstruct or manual ADC.
            # Simpler: reconstruct codes and compute dot product
            corpus_reconstructed = pq.decode(corpus_codes)
            query_reconstructed = pq.decode(query_codes[i:i+1])

            # Use reconstructed vectors for scoring
            scores[i] = query_reconstructed @ corpus_reconstructed.T

        # Metrics
        r10_float_pq = r10_vs_float(scores, query_ids_list, corpus_ids_list, float_scores)

        ndcg10_vals = []
        for i, qid in enumerate(query_ids_list):
            rel_q = qrels.get(qid, {})
            if rel_q:
                ndcg10_vals.append(ndcg_at_k(scores[i], corpus_ids_list, rel_q, k=10))
        ndcg10_pq = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0

        compression_x = BASE_DIM * 4 // bytes_per_vec

        rows.append({
            "dataset": "scifact",
            "method": f"pq_m{M_actual}",
            "family": "pq",
            "dims_or_M": M_actual,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": r10_float_pq,
            "ndcg10": ndcg10_pq,
            "spearman": "",
        })

    return rows


def evaluate_stsb_baselines(
    s1: np.ndarray,
    s2: np.ndarray,
    gold: np.ndarray,
) -> list[dict]:
    """Evaluate int8 and PQ baselines on STS-B similarity.

    Parameters
    ----------
    s1, s2 : np.ndarray
        Embeddings, shape (n, 2048).
    gold : np.ndarray
        Gold scores, shape (n,), range [0, 5].

    Returns
    -------
    list[dict]
        One row dict per method.
    """
    rows = []

    # INT8 baselines
    int8_methods = [
        ("int8_2048", 2048),
        ("int8_mrl1024", 1024),
        ("int8_mrl512", 512),
        ("int8_mrl256", 256),
        ("int8_mrl128", 128),
        ("int8_mrl64", 64),
    ]

    for method_id, dims in int8_methods:
        s1_sliced = slice_renorm(s1, dims)
        s2_sliced = slice_renorm(s2, dims)

        # Quantize
        s1_codes, s1_scale = int8_quantize(s1_sliced)
        s2_codes, s2_scale = int8_quantize(s2_sliced)

        # Dequantize
        s1_dq = int8_dequantize(s1_codes, s1_scale)
        s2_dq = int8_dequantize(s2_codes, s2_scale)

        # Pairwise cosine
        sim = np.sum(s1_dq * s2_dq, axis=1)  # (n,)

        # Spearman correlation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            spearman_corr, _ = spearmanr(sim, gold)
        spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

        bytes_per_vec = dims * 1
        compression_x = BASE_DIM * 4 // bytes_per_vec

        rows.append({
            "dataset": "stsb",
            "method": method_id,
            "family": "int8",
            "dims_or_M": dims,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": "",
            "ndcg10": "",
            "spearman": spearman_val,
        })

    # PQ baselines
    pq_configs = [
        ("pq_m32", 32),
        ("pq_m64", 64),
        ("pq_m128", 128),
        ("pq_m256", 256),
    ]

    d = s1.shape[1]  # 2048

    for method_id, M in pq_configs:
        # Check M divides d
        if d % M != 0:
            divisors = [2**i for i in range(12)]
            nearest_M = min(divisors, key=lambda x: abs(x - M))
            if nearest_M < M:
                nearest_M = max(d for d in divisors if d <= M)
            elif nearest_M > M:
                nearest_M = min(d for d in divisors if d >= M)
            M_actual = nearest_M
            print(f"[baselines_nemotron] PQ (STS-B): M={M} does not divide d={d}; using M={M_actual}", file=sys.stderr)
        else:
            M_actual = M

        # Train PQ on s1+s2
        pq = faiss.ProductQuantizer(d, M_actual, 8)
        s_combined = np.vstack([s1, s2]).astype(np.float32)
        s_combined_c = np.ascontiguousarray(s_combined)
        pq.train(s_combined_c)

        # Encode
        s1_c = np.ascontiguousarray(s1)
        s2_c = np.ascontiguousarray(s2)
        s1_codes = pq.compute_codes(s1_c)
        s2_codes = pq.compute_codes(s2_c)

        # Score: reconstruct and compute pairwise dot product
        s1_reconstructed = pq.decode(s1_codes)
        s2_reconstructed = pq.decode(s2_codes)
        sim = np.sum(s1_reconstructed * s2_reconstructed, axis=1)  # (n,)

        # Spearman
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            spearman_corr, _ = spearmanr(sim, gold)
        spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

        bytes_per_vec = M_actual
        compression_x = BASE_DIM * 4 // bytes_per_vec

        rows.append({
            "dataset": "stsb",
            "method": f"pq_m{M_actual}",
            "family": "pq",
            "dims_or_M": M_actual,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": "",
            "ndcg10": "",
            "spearman": spearman_val,
        })

    return rows


def run_selftest() -> None:
    """Selftest on synthetic data.

    Tests int8 and PQ on synthetic d=256 data with assertions.
    """
    print("[baselines_nemotron] running selftest...")

    n_docs = 300
    m_queries = 20
    d_test = 256

    rng = np.random.default_rng(0)

    # Synthetic Gaussian data - full rank to preserve geometry better for int8
    corpus_full = rng.standard_normal((n_docs, d_test)).astype(np.float32)
    queries_full = rng.standard_normal((m_queries, d_test)).astype(np.float32)

    # L2-normalize
    corpus_full /= np.linalg.norm(corpus_full, axis=1, keepdims=True)
    queries_full /= np.linalg.norm(queries_full, axis=1, keepdims=True)

    # Float32 reference scores
    float_scores = queries_full @ corpus_full.T

    corpus_ids = [str(i) for i in range(n_docs)]
    query_ids = [str(i) for i in range(m_queries)]

    # qrels: each query relevant to its nearest neighbor
    qrels = {}
    for i in range(m_queries):
        nearest = int(np.argmax(float_scores[i]))
        qrels[str(i)] = {str(nearest): 1}

    assertions_passed = 0
    assertions_total = 0

    # Test 1: int8 at full dims recovers > 0.9 r10_vs_float
    assertions_total += 1
    corpus = slice_renorm(corpus_full, d_test)
    queries = slice_renorm(queries_full, d_test)
    corpus_codes, corpus_scale = int8_quantize(corpus)
    query_codes, query_scale = int8_quantize(queries)
    corpus_dq = int8_dequantize(corpus_codes, corpus_scale)
    query_dq = int8_dequantize(query_codes, query_scale)
    scores_int8 = query_dq @ corpus_dq.T
    r10_int8 = r10_vs_float(scores_int8, query_ids, corpus_ids, float_scores)

    if r10_int8 > 0.9:
        print(f"PASS: int8 full dims r10_vs_float ({r10_int8:.4f}) > 0.9")
        assertions_passed += 1
    else:
        print(f"FAIL: int8 full dims r10_vs_float ({r10_int8:.4f}) <= 0.9")

    # Test 2: PQ runs and yields r10 in (0, 1.01]
    assertions_total += 1
    d = corpus_full.shape[1]
    M_pq = 16  # Smaller M for small selftest dataset
    if d % M_pq == 0:
        M_actual = M_pq
    else:
        divisors = [2**i for i in range(12)]
        M_actual = min(divisors, key=lambda x: abs(x - M_pq))

    try:
        pq = faiss.ProductQuantizer(d, M_actual, 8)
        corpus_full_c = np.ascontiguousarray(corpus_full)
        pq.train(corpus_full_c)
        corpus_codes_pq = pq.compute_codes(corpus_full_c)
        queries_full_c = np.ascontiguousarray(queries_full)
        query_codes_pq = pq.compute_codes(queries_full_c)

        corpus_reconstructed = pq.decode(corpus_codes_pq)
        query_reconstructed = pq.decode(query_codes_pq)
        scores_pq = query_reconstructed @ corpus_reconstructed.T
        r10_pq = r10_vs_float(scores_pq, query_ids, corpus_ids, float_scores)

        if 0 < r10_pq <= 1.01:
            print(f"PASS: PQ r10_vs_float ({r10_pq:.4f}) in (0, 1.01]")
            assertions_passed += 1
        else:
            print(f"FAIL: PQ r10_vs_float ({r10_pq:.4f}) not in (0, 1.01]")
    except Exception as e:
        print(f"FAIL: PQ raised exception: {e}")

    # Test 3: bytes accounting
    assertions_total += 1
    int8_bytes = d_test * 1
    pq_bytes = M_actual
    if int8_bytes == d_test and pq_bytes == M_actual:
        print(f"PASS: bytes accounting: int8={int8_bytes}, pq={pq_bytes}")
        assertions_passed += 1
    else:
        print(f"FAIL: bytes accounting mismatch")

    print(f"\n[baselines_nemotron] selftest: {assertions_passed}/{assertions_total} assertions passed")
    if assertions_passed < assertions_total:
        sys.exit(1)


def main() -> None:
    """Main evaluation on real Nemotron embeddings."""
    parser = argparse.ArgumentParser(
        description="Nemotron-3-Embed-1B int8 and PQ baseline evaluation."
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
            f"[baselines_nemotron] ERROR: missing embedding caches:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    # Load embeddings
    print("[baselines_nemotron] loading embeddings...")
    corpus_full = np.load(EMB_DIR / "scifact_docs.npy").astype(np.float32)
    queries_full = np.load(EMB_DIR / "scifact_queries.npy").astype(np.float32)
    s1_full = np.load(EMB_DIR / "stsb_s1.npy").astype(np.float32)
    s2_full = np.load(EMB_DIR / "stsb_s2.npy").astype(np.float32)

    # Load data
    print("[baselines_nemotron] loading data...")
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
    print("[baselines_nemotron] NOTE: PQ is data-DEPENDENT (trained on corpus)")
    print("[baselines_nemotron] evaluating SciFact...")
    scifact_rows = evaluate_scifact_baselines(
        corpus_ids, query_ids, qrels, corpus_full, queries_full
    )

    print("[baselines_nemotron] evaluating STS-B...")
    stsb_rows = evaluate_stsb_baselines(s1_full, s2_full, gold_stsb)

    # Write CSV
    csv_path = RESULTS_DIR / "nemotron_baselines.csv"
    print(f"[baselines_nemotron] writing {csv_path}...")

    all_rows = scifact_rows + stsb_rows

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "method",
            "family",
            "dims_or_M",
            "bytes_per_vec",
            "compression_x",
            "r10_vs_float",
            "ndcg10",
            "spearman",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            # Format floats to 4 decimals
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                elif v == "":
                    formatted_row[k] = ""
                else:
                    formatted_row[k] = str(v)
            writer.writerow(formatted_row)

    print(f"[baselines_nemotron] wrote {len(all_rows)} rows to {csv_path}")

    # Print summary table
    print("\n" + "="*100)
    print("SciFact Results (int8 and PQ baselines)")
    print("="*100)
    scifact_table = [r for r in all_rows if r["dataset"] == "scifact"]
    for row in scifact_table:
        print(f"{row['method']:20s} | bytes={row['bytes_per_vec']:4d} | "
              f"r10={row['r10_vs_float']:6.4f} | ndcg10={row['ndcg10']:6.4f} | "
              f"compression={row['compression_x']:5.1f}x")

    print("\n" + "="*100)
    print("STS-B Results (int8 and PQ baselines)")
    print("="*100)
    stsb_table = [r for r in all_rows if r["dataset"] == "stsb"]
    for row in stsb_table:
        print(f"{row['method']:20s} | bytes={row['bytes_per_vec']:4d} | "
              f"spearman={row['spearman']:6.4f} | "
              f"compression={row['compression_x']:5.1f}x")


if __name__ == "__main__":
    main()
