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
where $SCRATCH resolves via bench/nemotron_paths.py:
  $NEMOTRON_SCRATCH, else $SCRATCH, else bench/.cache/nemotron

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
from remax.packing import stable_top_k  # noqa: E402  (library's own O(n) top-k)

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

# Cache paths come from bench/nemotron_paths.py so every nemotron script
# resolves them the same way and all of them honour NEMOTRON_SCRATCH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemotron_paths import DATA_DIR, EMB_DIR, RESULTS_DIR, SCRATCH  # noqa: E402

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


# ── harness fairness ────────────────────────────────────────────────────────
#
# READ THIS BEFORE QUOTING ANY NUMBER OUT OF THIS FILE.
#
# The original harness did not give the two arms the same treatment, and the
# mismatch ran entirely one way — in the baseline's favour:
#
#   float32:  scores = queries @ corpus.T          ONE batched GEMM, all 300
#             np.argsort(-scores, axis=1)[:, :k]   ONE batched sort
#
#   remax:    for i in range(m):                   a Python loop, 300 calls
#                 hamming_distances(codes, q[i])
#                 np.argsort(distances)[:k]        a FULL O(n log n) sort,
#                                                  per query
#
# Three separate advantages, none of them a property of either algorithm:
#
#   1. The float32 arm amortises Python and dispatch overhead over 300
#      queries; the remax arm pays it 300 times.
#   2. BLAS on a (300, d) x (d, n) GEMM blocks the corpus and reuses it in
#      cache across all 300 queries. A per-query loop rereads the index 300
#      times.
#   3. remax was denied its own top-k. The library ships stable_top_k, an
#      argpartition-based O(n) selector; the harness called np.argsort, which
#      sorts all n. At n=5183, k=10 that is most of the remaining work.
#
# `mode` is now an explicit axis: both arms implement both modes, and both use
# an O(n) selector. The batched/per-query gap is a real and interesting number
# — it is what tells you whether your workload should batch — but it belongs
# to the harness, so it has to be visible rather than baked into one arm.


def _topk_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k by descending score with O(n) selection.

    The float32 counterpart of remax's ``stable_top_k``, so neither arm is
    left holding a full sort while the other is not.
    """
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    if scores.ndim == 1:
        return part[np.argsort(-scores[part])]
    rows = np.arange(scores.shape[0])[:, None]
    return np.take_along_axis(
        part, np.argsort(-scores[rows, part], axis=1), axis=1
    )


def benchmark_float32_matmul(
    corpus: np.ndarray, queries: np.ndarray, k: int, mode: str = "batched"
) -> tuple[float, float, float]:
    """Benchmark float32 matmul retrieval.

    mode="batched"    one GEMM over all m queries — throughput-shaped
    mode="per_query"  one query at a time — latency-shaped, and the shape the
                      remax arm was measured in before this became an axis

    Returns: (per_query_ms_median, per_query_ms_best, queries_per_sec)
    """
    m = queries.shape[0]

    if mode == "batched":
        def run():
            scores = queries @ corpus.T
            _topk_desc(scores, k)
    elif mode == "per_query":
        def run():
            for i in range(m):
                scores = queries[i] @ corpus.T
                _topk_desc(scores, k)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    run()  # warmup

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

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
    corpus: np.ndarray, queries: np.ndarray, k: int, k_stack: int = 1,
    mode: str = "batched",
) -> tuple[float, float, float]:
    """Benchmark the remax Hamming scan.

    Same two modes as the float32 arm, and ``stable_top_k`` rather than
    ``np.argsort`` — the library's own O(n) selector, which the original
    harness did not call. See the "harness fairness" note above.

    ``mode`` is accepted for symmetry with the float32 arm but both values
    time the *same* work, and that is the point rather than an oversight:
    **remax has no batched Hamming kernel.** ``search`` loops over queries in
    Python whichever way you call it (an m-way SIMD popcount is explicitly
    post-v0.1.0). Reporting a "batched" remax number that merely moved the
    loop inside the library would manufacture a distinction that does not
    exist in the code — and calling ``quant.search(queries, ...)`` here would
    be worse than that, because it drags a (m, d) x (d, d) rotation GEMM for
    all m queries inside the timer while the float32 arm has no encode step
    at all.

    Query encoding is therefore excluded from the timed region, as in the
    original harness, so these numbers stay comparable to the published ones.
    That exclusion favours remax and is stated in NEMOTRON_1BIT.md.
    """
    if mode not in ("batched", "per_query"):
        raise ValueError(f"unknown mode {mode!r}")

    d = corpus.shape[1]
    m = queries.shape[0]

    if k_stack == 1:
        quant = SignBitQuantizer(d=d, seed=QUANTIZER_SEED)
    else:
        quant = StackedSignBitQuantizer(d=d, k=k_stack, seed=QUANTIZER_SEED)
    corpus_codes = quant.encode(corpus)
    query_codes = quant.encode(queries)

    scratch = np.empty(corpus_codes.shape[0], dtype=np.int32)

    def run():
        for i in range(m):
            distances = hamming_distances(
                corpus_codes, query_codes[i], out=scratch
            )
            stable_top_k(distances, k)

    run()  # warmup

    times = []
    for _ in range(NUM_REPEATS):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

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
