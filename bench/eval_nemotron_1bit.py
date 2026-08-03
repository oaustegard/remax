"""Nemotron-3-Embed-1B 1-bit evaluation for remax.

Measures retrieval and similarity quality when Nemotron 3 2048-dim embeddings
are quantized to 1-bit with the remax library, compared to float32 baselines
(full and Matryoshka slices).

Data sources (embedding caches + qrels/gold):
  - $SCRATCH/emb/{scifact_docs,scifact_queries,stsb_s1,stsb_s2}.npy
  - $SCRATCH/data/{scifact_subset,stsb_test}.json
where $SCRATCH resolves via bench/nemotron_paths.py:
  $NEMOTRON_SCRATCH, else $SCRATCH, else bench/.cache/nemotron

Outputs:
  - /home/user/remax/bench/results/nemotron_1bit.csv

Usage
-----
    python bench/eval_nemotron_1bit.py --selftest
    python bench/eval_nemotron_1bit.py              # runs on caches

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

# Import remax — match eval_beir.py pattern
try:
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances
except ImportError:
    # Fallback: add src/ to path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances


# Cache paths come from bench/nemotron_paths.py so every nemotron script
# resolves them the same way and all of them honour NEMOTRON_SCRATCH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemotron_paths import DATA_DIR, EMB_DIR, RESULTS_DIR, SCRATCH  # noqa: E402

# Contract: base embedding dimension
BASE_DIM = 2048

# Method grid from contract
METHODS = [
    # (method_id, dims, k_stack, description)
    ("f32_2048", 2048, 1, "float32 full"),
    ("f32_mrl256", 256, 1, "float32 MRL slice 256"),
    ("f32_mrl128", 128, 1, "float32 MRL slice 128"),
    ("f32_mrl64", 64, 1, "float32 MRL slice 64"),
    ("bit1_2048", 2048, 1, "SignBitQuantizer d=2048 seed=0"),
    ("bit1_stack2", 2048, 2, "StackedSignBitQuantizer d=2048 k=2 seed=0"),
    ("bit1_stack4", 2048, 4, "StackedSignBitQuantizer d=2048 k=4 seed=0"),
    ("bit1_mrl1024", 1024, 1, "slice 1024 + SignBitQuantizer d=1024 seed=0"),
    ("bit1_mrl512", 512, 1, "slice 512 + SignBitQuantizer d=512 seed=0"),
    ("bit1_mrl256", 256, 1, "slice 256 + SignBitQuantizer d=256 seed=0"),
]

QUANTIZER_SEED = 0


def slice_renorm(emb: np.ndarray, d: int) -> np.ndarray:
    """Slice to first d dims and re-L2-normalize rows.

    Parameters
    ----------
    emb : np.ndarray, shape (n, original_d), dtype float32
    d : int
        Target dimension.

    Returns
    -------
    np.ndarray, shape (n, d), dtype float32, L2-normalized rows
    """
    if d == emb.shape[1]:
        return emb.astype(np.float32, copy=False)
    out = emb[:, :d].astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def r10_vs_float(
    query_scores: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    float_scores: np.ndarray,
) -> float:
    """R@10 overlap with float32-full cosine top-10.

    For each query, compute the intersection of the method's top-10 indices
    with the float32 top-10 indices (breaking ties stably), then average
    the overlap fraction.

    Parameters
    ----------
    query_scores : np.ndarray, shape (m, n)
        Higher-is-better similarity scores (one row per query).
    query_ids : list[str]
        Query IDs (unused, for future extensibility).
    corpus_ids : list[str]
        Corpus IDs (unused, for future extensibility).
    float_scores : np.ndarray, shape (m, n)
        Ground-truth float32 scores (for computing top-10).

    Returns
    -------
    float
        Mean overlap fraction, in [0, 1].
    """
    m, n = query_scores.shape
    k = 10
    overlaps = []

    # Compute float32 top-10 (ground truth)
    float_topk_idx = np.argsort(-float_scores, axis=1, kind='stable')[:, :k]

    # Compute method top-10
    method_topk_idx = np.argsort(-query_scores, axis=1, kind='stable')[:, :k]

    for i in range(m):
        float_set = set(float_topk_idx[i])
        method_set = set(method_topk_idx[i])
        overlap = len(float_set & method_set) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def ndcg_at_k(
    scores_q: np.ndarray,
    corpus_ids: list[str],
    rel_q: dict[str, int],
    k: int = 10,
) -> float:
    """nDCG@k for one query.

    Parameters
    ----------
    scores_q : np.ndarray, shape (n,)
        Higher-is-better scores for all corpus items.
    corpus_ids : list[str]
        Corpus IDs, aligned with scores_q.
    rel_q : dict[str, int]
        Query's relevance dict (corpus_id -> relevance grade).
    k : int
        Cutoff.

    Returns
    -------
    float
        nDCG@k, in [0, 1].
    """
    if not rel_q:
        return 0.0

    # Top-k indices
    top = np.argsort(-scores_q, kind='stable')[:k]

    # DCG
    dcg = 0.0
    for rank, idx in enumerate(top, start=1):
        cid = corpus_ids[idx]
        gain = rel_q.get(cid, 0)
        if gain > 0:
            dcg += gain / np.log2(rank + 1)

    # IDCG: sorted relevances from qrels, capped at k
    ideal_rels = sorted(rel_q.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 1) for r, g in enumerate(ideal_rels, start=1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_scifact(
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_full: np.ndarray,
    queries_full: np.ndarray,
) -> list[dict]:
    """Evaluate all methods on SciFact retrieval.

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
        One row dict per method.
    """
    rows = []
    query_ids_list = list(query_ids) if not isinstance(query_ids, list) else query_ids
    corpus_ids_list = list(corpus_ids) if not isinstance(corpus_ids, list) else corpus_ids

    # Precompute float32 top-10 for r10_vs_float metric
    float_scores = queries_full @ corpus_full.T  # (m, n)
    float_topk_idx = np.argsort(-float_scores, axis=1, kind='stable')[:, :10]

    for method_id, dims, k_stack, desc in METHODS:
        # Slice and optionally quantize
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)

        if method_id.startswith("f32"):
            # Float32 baseline
            scores = queries @ corpus.T  # (m, n)
            bytes_per_vec = dims * 4

        else:
            # Bit-quantized method
            from remax import hamming_distances

            if k_stack == 1:
                # Plain SignBitQuantizer
                q = SignBitQuantizer(d=dims, seed=QUANTIZER_SEED)
                corpus_codes = q.encode(corpus)
                bytes_per_vec = corpus_codes.shape[1]

                # Compute -Hamming distance scores (higher = more similar)
                query_codes = q.encode(queries)
                scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
                for i in range(queries.shape[0]):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])

            else:
                # StackedSignBitQuantizer
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=QUANTIZER_SEED)
                corpus_codes = sq.encode(corpus)
                bytes_per_vec = corpus_codes.shape[1]

                # Compute -Hamming distance scores (higher = more similar)
                query_codes = sq.encode(queries)
                scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
                for i in range(queries.shape[0]):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])

        # Compute metrics
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
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": r10_float,
            "ndcg10": ndcg10,
            "spearman": "",
        })

    return rows


def evaluate_stsb(
    s1: np.ndarray,
    s2: np.ndarray,
    gold: np.ndarray,
) -> list[dict]:
    """Evaluate all methods on STS-B similarity.

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

    for method_id, dims, k_stack, desc in METHODS:
        # Slice
        s1_sliced = slice_renorm(s1, dims)
        s2_sliced = slice_renorm(s2, dims)

        if method_id.startswith("f32"):
            # Float32 baseline: pairwise cosine
            sim = np.sum(s1_sliced * s2_sliced, axis=1)  # (n,)
            bytes_per_vec = dims * 4

        else:
            # Bit-quantized method: pairwise negative Hamming
            from remax import hamming_distances

            if k_stack == 1:
                q = SignBitQuantizer(d=dims, seed=QUANTIZER_SEED)
                s1_codes = q.encode(s1_sliced)
                s2_codes = q.encode(s2_sliced)
            else:
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=QUANTIZER_SEED)
                s1_codes = sq.encode(s1_sliced)
                s2_codes = sq.encode(s2_sliced)

            # Pairwise Hamming distance (negated for higher-is-better)
            sim = np.array(
                [-hamming_distances(s2_codes, s1_codes[i])[i] for i in range(len(s1_codes))]
            )
            bytes_per_vec = s1_codes.shape[1]

        # Spearman correlation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            spearman_corr, _ = spearmanr(sim, gold)
        spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

        compression_x = BASE_DIM * 4 // bytes_per_vec

        rows.append({
            "dataset": "stsb",
            "method": method_id,
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": "",
            "ndcg10": "",
            "spearman": spearman_val,
        })

    return rows


def run_selftest() -> None:
    """Selftest on synthetic data.

    Generates n=200 docs, m=20 queries, d=256 with scaled MRL grid
    (slices 128/64/32, stacks k=2,4). Checks positive-control assertions.
    """
    print("[eval_nemotron_1bit] running selftest...")

    n_docs = 200
    m_queries = 20
    d_test = 256

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
    # Create 100 pairs of vectors
    s1_raw = rng.standard_normal((100, subdim)) @ basis.T
    s2_raw = rng.standard_normal((100, subdim)) @ basis.T
    s1_pairs = s1_raw.astype(np.float32)
    s2_pairs = s2_raw.astype(np.float32)
    s1_pairs /= np.linalg.norm(s1_pairs, axis=1, keepdims=True)
    s2_pairs /= np.linalg.norm(s2_pairs, axis=1, keepdims=True)
    gold_pairs = np.sum(s1_pairs * s2_pairs, axis=1) * 5  # scale to [0, 5]

    # Adapted METHODS grid for selftest (scaled down)
    selftest_methods = [
        ("f32_256", 256, 1),
        ("f32_mrl128", 128, 1),
        ("f32_mrl64", 64, 1),
        ("bit1_256", 256, 1),
        ("bit1_stack2", 256, 2),
        ("bit1_stack4", 256, 4),
        ("bit1_mrl128", 128, 1),
    ]

    # Evaluate SciFact
    scifact_rows = []
    float_scores = queries_full @ corpus_full.T

    for method_id, dims, k_stack in selftest_methods:
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)

        if method_id.startswith("f32"):
            scores = queries @ corpus.T
            bytes_per_vec = dims * 4
        else:
            from remax import hamming_distances

            if k_stack == 1:
                q = SignBitQuantizer(d=dims, seed=QUANTIZER_SEED)
                corpus_codes = q.encode(corpus)
                query_codes = q.encode(queries)
                scores = np.empty((m_queries, n_docs), dtype=np.float64)
                for i in range(m_queries):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])
            else:
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=QUANTIZER_SEED)
                corpus_codes = sq.encode(corpus)
                query_codes = sq.encode(queries)
                scores = np.empty((m_queries, n_docs), dtype=np.float64)
                for i in range(m_queries):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])
            bytes_per_vec = corpus_codes.shape[1]

        r10 = r10_vs_float(scores, query_ids, corpus_ids, float_scores)

        compression_x = d_test * 4 // bytes_per_vec

        scifact_rows.append({
            "method": method_id,
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "compression_x": compression_x,
            "r10_vs_float": r10,
        })

    # Evaluate STS-B
    stsb_rows = []
    for method_id, dims, k_stack in selftest_methods:
        s1 = slice_renorm(s1_pairs, dims)
        s2 = slice_renorm(s2_pairs, dims)

        if method_id.startswith("f32"):
            sim = np.sum(s1 * s2, axis=1)
            bytes_per_vec = dims * 4
        else:
            from remax import hamming_distances

            if k_stack == 1:
                q = SignBitQuantizer(d=dims, seed=QUANTIZER_SEED)
                s1_codes = q.encode(s1)
                s2_codes = q.encode(s2)
            else:
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=QUANTIZER_SEED)
                s1_codes = sq.encode(s1)
                s2_codes = sq.encode(s2)

            sim = np.array(
                [-hamming_distances(s2_codes, s1_codes[i])[i] for i in range(len(s1_codes))]
            )
            bytes_per_vec = s1_codes.shape[1]

        spearman_val, _ = spearmanr(sim, gold_pairs)
        stsb_rows.append({
            "method": method_id,
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "spearman": spearman_val,
        })

    # Assertions
    assertions_passed = 0
    assertions_total = 0

    # (a) f32 full row has r10_vs_float == 1.0
    assertions_total += 1
    f32_256_r10 = [r for r in scifact_rows if r['method'] == 'f32_256'][0]['r10_vs_float']
    if f32_256_r10 == 1.0:
        print("PASS: f32 full r10_vs_float == 1.0")
        assertions_passed += 1
    else:
        print(f"FAIL: f32 full r10_vs_float == {f32_256_r10}, expected 1.0")

    # (b) f32 full spearman > 0.99
    assertions_total += 1
    f32_256_spearman = [r for r in stsb_rows if r['method'] == 'f32_256'][0]['spearman']
    if f32_256_spearman > 0.99:
        print(f"PASS: f32 full spearman == {f32_256_spearman:.4f} > 0.99")
        assertions_passed += 1
    else:
        print(f"FAIL: f32 full spearman == {f32_256_spearman:.4f}, expected > 0.99")

    # (c) every bit method has r10_vs_float in (0, 1.01]
    assertions_total += 1
    bit_methods_scifact = [r for r in scifact_rows if r['method'].startswith('bit')]
    all_in_range = all(0 < r['r10_vs_float'] <= 1.01 for r in bit_methods_scifact)
    if all_in_range:
        print(f"PASS: all {len(bit_methods_scifact)} bit methods have r10_vs_float in (0, 1.01]")
        assertions_passed += 1
    else:
        bad = [r['method'] for r in bit_methods_scifact if not (0 < r['r10_vs_float'] <= 1.01)]
        print(f"FAIL: bit methods {bad} have r10_vs_float outside (0, 1.01]")

    # (d) bytes_per_vec halves down MRL ladder
    assertions_total += 1
    f32_256_bpv = [r for r in scifact_rows if r['method'] == 'f32_256'][0]['bytes_per_vec']
    f32_128_bpv = [r for r in scifact_rows if r['method'] == 'f32_mrl128'][0]['bytes_per_vec']
    f32_64_bpv = [r for r in scifact_rows if r['method'] == 'f32_mrl64'][0]['bytes_per_vec']
    bytes_halve = (f32_256_bpv == 2 * f32_128_bpv and f32_128_bpv == 2 * f32_64_bpv)
    if bytes_halve:
        print(f"PASS: bytes_per_vec halves: {f32_256_bpv} → {f32_128_bpv} → {f32_64_bpv}")
        assertions_passed += 1
    else:
        print(f"FAIL: bytes_per_vec does not halve: {f32_256_bpv}, {f32_128_bpv}, {f32_64_bpv}")

    # (e) stacked k=4 r10_vs_float >= plain 1-bit r10_vs_float
    assertions_total += 1
    bit_k4_r10 = [r for r in scifact_rows if r['method'] == 'bit1_stack4'][0]['r10_vs_float']
    bit_k1_r10 = [r for r in scifact_rows if r['method'] == 'bit1_256'][0]['r10_vs_float']
    if bit_k4_r10 >= bit_k1_r10:
        print(f"PASS: stacked k=4 r10 ({bit_k4_r10:.4f}) >= 1-bit r10 ({bit_k1_r10:.4f})")
        assertions_passed += 1
    else:
        print(f"FAIL: stacked k=4 r10 ({bit_k4_r10:.4f}) < 1-bit r10 ({bit_k1_r10:.4f})")

    print(f"\n[eval_nemotron_1bit] selftest: {assertions_passed}/{assertions_total} assertions passed")
    if assertions_passed < assertions_total:
        sys.exit(1)


def main() -> None:
    """Main evaluation on real Nemotron embeddings."""
    parser = argparse.ArgumentParser(
        description="Nemotron-3-Embed-1B 1-bit evaluation for remax."
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
            f"[eval_nemotron_1bit] ERROR: missing embedding caches:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        print(
            "\nRun Agent A (embed_nemotron.py) first to generate these caches.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load embeddings
    print("[eval_nemotron_1bit] loading embeddings...")
    corpus_full = np.load(EMB_DIR / "scifact_docs.npy").astype(np.float32)
    queries_full = np.load(EMB_DIR / "scifact_queries.npy").astype(np.float32)
    s1_full = np.load(EMB_DIR / "stsb_s1.npy").astype(np.float32)
    s2_full = np.load(EMB_DIR / "stsb_s2.npy").astype(np.float32)

    # Load data
    print("[eval_nemotron_1bit] loading data...")
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
    print("[eval_nemotron_1bit] evaluating SciFact...")
    scifact_rows = evaluate_scifact(
        corpus_ids, query_ids, qrels, corpus_full, queries_full
    )

    print("[eval_nemotron_1bit] evaluating STS-B...")
    stsb_rows = evaluate_stsb(s1_full, s2_full, gold_stsb)

    # Write CSV
    csv_path = RESULTS_DIR / "nemotron_1bit.csv"
    print(f"[eval_nemotron_1bit] writing {csv_path}...")

    all_rows = scifact_rows + stsb_rows

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "method",
            "dims",
            "k_stack",
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

    print(f"[eval_nemotron_1bit] wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
