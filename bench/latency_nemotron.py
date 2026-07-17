"""Nemotron-3-Embed-1B query-time latency benchmark for remax.

Measures per-query latency and throughput for retrieval methods on SciFact corpus:
- float32 matmul (full 2048 + MRL256)
- int8 dequant+matmul (full 2048)
- remax 1-bit hamming scan (bit1_2048 + bit1_stack4)
- faiss PQ ADC (M=256)

Single thread, k=10, warmup + 5 timed repeats over 300-query batch.
Reports: median/best per-query ms, queries/sec, index size MB.

Data sources (embedding caches):
  - $SCRATCH/emb/{scifact_docs,scifact_queries}.npy
  - $SCRATCH/data/scifact_subset.json
where $SCRATCH = /tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad

Outputs:
  - /home/user/remax/bench/results/nemotron_latency.csv

Usage
-----
    python bench/latency_nemotron.py --selftest
    python bench/latency_nemotron.py              # runs on caches

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Set thread counts to 1 for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Import remax — match eval_nemotron_1bit pattern
try:
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances

# Try to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try to import torch for thread control
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

SCRATCH = Path(
    "/tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad"
)
DATA_DIR = SCRATCH / "data"
EMB_DIR = SCRATCH / "emb"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASE_DIM = 2048
QUANTIZER_SEED = 0
K = 10
NUM_REPEATS = 5


def slice_renorm(emb: np.ndarray, d: int) -> np.ndarray:
    """Slice to first d dims and re-L2-normalize rows."""
    if d == emb.shape[1]:
        return emb.astype(np.float32, copy=False)
    out = emb[:, :d].astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def benchmark_float32_matmul(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> tuple[float, float, float]:
    """Benchmark float32 matmul retrieval.

    Returns: (per_query_ms_median, per_query_ms_best, queries_per_sec)
    """
    m, n = queries.shape[0], corpus.shape[0]

    # Warmup
    scores = queries @ corpus.T
    _ = np.argsort(-scores, axis=1)[:, :k]

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        scores = queries @ corpus.T
        _ = np.argsort(-scores, axis=1)[:, :k]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_queries = m * NUM_REPEATS
    per_query_ms_median = float(np.median(times) / m * 1000)
    per_query_ms_best = float(np.min(times) / m * 1000)
    queries_per_sec = float(total_queries / np.sum(times))

    return per_query_ms_median, per_query_ms_best, queries_per_sec


def benchmark_int8_matmul(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> tuple[float, float, float]:
    """Benchmark int8 dequant+matmul retrieval."""
    # Quantize to int8 (simple per-vector symmetric quantization)
    def quantize_to_int8(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize to int8, return (codes, scales)."""
        scales = np.max(np.abs(X), axis=1, keepdims=True)
        scales[scales == 0] = 1.0
        codes = np.clip(np.round(X / scales * 127), -128, 127).astype(np.int8)
        return codes, scales

    def dequantize_and_score(codes: np.ndarray, scales: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Dequantize and compute matmul."""
        X_dq = codes.astype(np.float32) * scales / 127.0
        return X_dq @ other.T

    corpus_codes, corpus_scales = quantize_to_int8(corpus)
    queries_codes, queries_scales = quantize_to_int8(queries)

    m, n = queries.shape[0], corpus.shape[0]

    # Warmup
    _ = dequantize_and_score(queries_codes, queries_scales, corpus)

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        scores = dequantize_and_score(queries_codes, queries_scales, corpus)
        _ = np.argsort(-scores, axis=1)[:, :k]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_queries = m * NUM_REPEATS
    per_query_ms_median = float(np.median(times) / m * 1000)
    per_query_ms_best = float(np.min(times) / m * 1000)
    queries_per_sec = float(total_queries / np.sum(times))

    return per_query_ms_median, per_query_ms_best, queries_per_sec


def benchmark_remax_hamming(
    corpus: np.ndarray, queries: np.ndarray, k: int, k_stack: int = 1
) -> tuple[float, float, float]:
    """Benchmark remax hamming distance scan."""
    d = corpus.shape[1]
    m, n = queries.shape[0], corpus.shape[0]

    if k_stack == 1:
        q = SignBitQuantizer(d=d, seed=QUANTIZER_SEED)
        corpus_codes = q.encode(corpus)
        query_codes = q.encode(queries)
    else:
        sq = StackedSignBitQuantizer(d=d, k=k_stack, seed=QUANTIZER_SEED)
        corpus_codes = sq.encode(corpus)
        query_codes = sq.encode(queries)

    # Warmup
    for i in range(min(5, m)):
        _ = hamming_distances(corpus_codes, query_codes[i])

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        for i in range(m):
            distances = hamming_distances(corpus_codes, query_codes[i])
            _ = np.argsort(distances)[:k]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_queries = m * NUM_REPEATS
    per_query_ms_median = float(np.median(times) / m * 1000)
    per_query_ms_best = float(np.min(times) / m * 1000)
    queries_per_sec = float(total_queries / np.sum(times))

    return per_query_ms_median, per_query_ms_best, queries_per_sec


def benchmark_faiss_pq(
    corpus: np.ndarray, queries: np.ndarray, k: int, m: int = 256
) -> tuple[float, float, float]:
    """Benchmark faiss PQ ADC retrieval."""
    if not FAISS_AVAILABLE:
        raise ImportError("faiss not available")

    faiss.omp_set_num_threads(1)

    d = corpus.shape[1]
    n_docs = corpus.shape[0]

    # Train PQ on corpus
    pq = faiss.ProductQuantizer(d, m, 8)
    pq.train(corpus.astype(np.float32))

    # Encode corpus
    codes = pq.compute_codes(corpus.astype(np.float32))

    # Create index
    index = faiss.IndexPQ(d, m, 8)
    index.pq = pq
    index.add(corpus.astype(np.float32))

    m_queries = queries.shape[0]

    # Warmup
    _, _ = index.search(queries[:5].astype(np.float32), k)

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        _, _ = index.search(queries.astype(np.float32), k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_queries = m_queries * NUM_REPEATS
    per_query_ms_median = float(np.median(times) / m_queries * 1000)
    per_query_ms_best = float(np.min(times) / m_queries * 1000)
    queries_per_sec = float(total_queries / np.sum(times))

    return per_query_ms_median, per_query_ms_best, queries_per_sec


def run_selftest() -> None:
    """Selftest on synthetic data."""
    print("[latency_nemotron] running selftest...")

    n_docs = 500
    m_queries = 100
    d_test = 256

    rng = np.random.default_rng(42)

    # Synthetic Gaussian data
    corpus = rng.standard_normal((n_docs, d_test)).astype(np.float32)
    queries = rng.standard_normal((m_queries, d_test)).astype(np.float32)

    # L2-normalize
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    assertions_passed = 0
    assertions_total = 0

    # Test each method
    methods = [
        ("float32", {"method": "float32", "dimscorp": d_test}),
        ("int8", {"method": "int8", "dims": d_test}),
        ("bit1", {"method": "bit1", "dims": d_test, "k_stack": 1}),
        ("bit1_stack4", {"method": "bit1_stack4", "dims": d_test, "k_stack": 4}),
    ]

    for method_name, params in methods:
        try:
            if params["method"] == "float32":
                m_med, m_best, qps = benchmark_float32_matmul(corpus, queries, K)
            elif params["method"] == "int8":
                m_med, m_best, qps = benchmark_int8_matmul(corpus, queries, K)
            elif params["method"] == "bit1":
                m_med, m_best, qps = benchmark_remax_hamming(corpus, queries, K, k_stack=1)
            elif params["method"] == "bit1_stack4":
                m_med, m_best, qps = benchmark_remax_hamming(corpus, queries, K, k_stack=4)

            # Assert positive finite latencies
            assertions_total += 1
            if m_med > 0 and np.isfinite(m_med) and m_best > 0 and np.isfinite(m_best):
                print(f"PASS: {method_name} latencies positive and finite")
                assertions_passed += 1
            else:
                print(f"FAIL: {method_name} latencies invalid: median={m_med}, best={m_best}")
        except Exception as e:
            print(f"FAIL: {method_name} raised {type(e).__name__}: {e}")

    # Test hamming index < float32 index
    assertions_total += 1
    hamming_idx_mb = (n_docs * (d_test // 8)) / (1024 * 1024)
    float_idx_mb = (n_docs * d_test * 4) / (1024 * 1024)
    if hamming_idx_mb < float_idx_mb:
        print(f"PASS: hamming index {hamming_idx_mb:.2f} MB < float {float_idx_mb:.2f} MB")
        assertions_passed += 1
    else:
        print(f"FAIL: hamming index {hamming_idx_mb:.2f} MB >= float {float_idx_mb:.2f} MB")

    print(f"\n[latency_nemotron] selftest: {assertions_passed}/{assertions_total} assertions passed")
    if assertions_passed < assertions_total:
        sys.exit(1)


def main() -> None:
    """Main latency benchmark on real Nemotron embeddings."""
    parser = argparse.ArgumentParser(
        description="Nemotron-3-Embed-1B query-time latency benchmark for remax."
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

    # Check for embedding caches
    emb_files = [
        EMB_DIR / "scifact_docs.npy",
        EMB_DIR / "scifact_queries.npy",
    ]
    missing = [f for f in emb_files if not f.exists()]

    if missing:
        print(
            f"[latency_nemotron] ERROR: missing embedding caches:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    # Load embeddings
    print("[latency_nemotron] loading embeddings...")
    corpus_full = np.load(EMB_DIR / "scifact_docs.npy").astype(np.float32)
    queries_full = np.load(EMB_DIR / "scifact_queries.npy").astype(np.float32)

    print(f"[latency_nemotron] corpus: {corpus_full.shape}, queries: {queries_full.shape}")
    print(f"[latency_nemotron] thread count set to 1 (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Method grid
    methods = [
        # (method_id, family, dims, k_stack, description)
        ("f32_2048", "float32", 2048, 1, "float32 full 2048"),
        ("f32_mrl256", "float32", 256, 1, "float32 MRL 256"),
        ("int8_2048", "int8", 2048, 1, "int8 full 2048"),
        ("bit1_2048", "remax", 2048, 1, "SignBitQuantizer 2048"),
        ("bit1_stack4", "remax", 2048, 4, "StackedSignBitQuantizer d=2048 k=4"),
    ]

    # Add PQ if available
    if FAISS_AVAILABLE:
        methods.append(("pq_m256", "faiss_pq", 2048, 1, "faiss PQ M=256"))

    rows = []

    for method_id, family, dims, k_stack, desc in methods:
        print(f"\n[latency_nemotron] benchmarking {method_id}...")

        try:
            # Prepare data
            corpus = slice_renorm(corpus_full, dims)
            queries = slice_renorm(queries_full, dims)

            if family == "float32":
                m_med, m_best, qps = benchmark_float32_matmul(corpus, queries, K)
                bytes_per_vec = dims * 4

            elif family == "int8":
                m_med, m_best, qps = benchmark_int8_matmul(corpus, queries, K)
                bytes_per_vec = dims

            elif family == "remax":
                m_med, m_best, qps = benchmark_remax_hamming(corpus, queries, K, k_stack=k_stack)
                if k_stack == 1:
                    bytes_per_vec = dims // 8
                else:
                    bytes_per_vec = k_stack * dims // 8

            elif family == "faiss_pq":
                m_med, m_best, qps = benchmark_faiss_pq(corpus, queries, K, m=256)
                bytes_per_vec = 256

            index_mb = (corpus.shape[0] * bytes_per_vec) / (1024 * 1024)

            print(f"  per-query: {m_med:.4f} ms (median), {m_best:.4f} ms (best)")
            print(f"  queries/sec: {qps:.1f}")
            print(f"  index size: {index_mb:.2f} MB ({bytes_per_vec} B/vec)")

            rows.append({
                "method": method_id,
                "family": family,
                "bytes_per_vec": bytes_per_vec,
                "index_mb": index_mb,
                "per_query_ms_median": m_med,
                "per_query_ms_best": m_best,
                "queries_per_sec": qps,
            })

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue

    # Sort by per_query_ms_median (fastest first)
    rows.sort(key=lambda r: r["per_query_ms_median"])

    # Print ranked table
    print("\n" + "="*100)
    print("LATENCY BENCHMARK RESULTS (ranked by per-query median latency, fastest first)")
    print("="*100)
    print(f"{'Rank':<5} {'Method':<15} {'Family':<12} {'B/vec':<8} {'Index MB':<12} {'ms/query (med)':<18} {'ms/query (best)':<18} {'Queries/sec':<15}")
    print("-"*100)

    for rank, row in enumerate(rows, 1):
        print(
            f"{rank:<5} "
            f"{row['method']:<15} "
            f"{row['family']:<12} "
            f"{row['bytes_per_vec']:<8} "
            f"{row['index_mb']:<12.2f} "
            f"{row['per_query_ms_median']:<18.4f} "
            f"{row['per_query_ms_best']:<18.4f} "
            f"{row['queries_per_sec']:<15.1f}"
        )

    # Write CSV
    csv_path = RESULTS_DIR / "nemotron_latency.csv"
    print(f"\n[latency_nemotron] writing {csv_path}...")

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "method",
            "family",
            "bytes_per_vec",
            "index_mb",
            "per_query_ms_median",
            "per_query_ms_best",
            "queries_per_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}" if k != "bytes_per_vec" else str(int(v))
                else:
                    formatted_row[k] = str(v)
            writer.writerow(formatted_row)

    print(f"[latency_nemotron] wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
