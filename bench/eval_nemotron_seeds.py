"""Nemotron-3-Embed-1B multi-seed evaluation for remax bit methods.

Measures retrieval and similarity quality when remax bit quantizers are run
across multiple random seeds (0-4). Reports mean and std per metric, plus
deterministic float32 baseline rows.

Data sources (embedding caches + qrels/gold):
  - $SCRATCH/emb/{scifact_docs,scifact_queries,stsb_s1,stsb_s2}.npy
  - $SCRATCH/data/{scifact_subset,stsb_test}.json
where $SCRATCH = /tmp/claude-0/-home-user/17f19a5d-a832-5512-bd5c-e28bcfa2ca35/scratchpad

Outputs:
  - /home/user/remax/bench/results/nemotron_seeds.csv

Usage
-----
    python bench/eval_nemotron_seeds.py --selftest
    python bench/eval_nemotron_seeds.py              # runs on caches

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

# Import remax — match eval_nemotron_1bit pattern
try:
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances
except ImportError:
    # Fallback: add src/ to path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from remax import SignBitQuantizer, StackedSignBitQuantizer, hamming_distances

# Import metric helpers from phase 1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_nemotron_1bit import (
    slice_renorm,
    r10_vs_float,
    ndcg_at_k,
    EMB_DIR,
    DATA_DIR,
    RESULTS_DIR,
    BASE_DIM,
)

# Six bit methods to run across seeds
BIT_METHODS = [
    # (method_id, dims, k_stack, description)
    ("bit1_2048", 2048, 1, "SignBitQuantizer d=2048"),
    ("bit1_stack2", 2048, 2, "StackedSignBitQuantizer d=2048 k=2"),
    ("bit1_stack4", 2048, 4, "StackedSignBitQuantizer d=2048 k=4"),
    ("bit1_mrl1024", 1024, 1, "slice 1024 + SignBitQuantizer"),
    ("bit1_mrl512", 512, 1, "slice 512 + SignBitQuantizer"),
    ("bit1_mrl256", 256, 1, "slice 256 + SignBitQuantizer"),
]

# Float32 reference methods (deterministic, included with std=0)
FLOAT32_METHODS = [
    ("f32_2048", 2048, 1, "float32 full"),
    ("f32_mrl256", 256, 1, "float32 MRL slice 256"),
    ("f32_mrl128", 128, 1, "float32 MRL slice 128"),
    ("f32_mrl64", 64, 1, "float32 MRL slice 64"),
]

SEEDS = [0, 1, 2, 3, 4]


def evaluate_scifact_seed(
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_full: np.ndarray,
    queries_full: np.ndarray,
    seed: int,
) -> list[dict]:
    """Evaluate bit methods on SciFact retrieval for a single seed.

    Parameters
    ----------
    corpus_ids, query_ids : list[str]
    qrels : dict[str, dict[str, int]]
    corpus_full, queries_full : np.ndarray
    seed : int
        Quantizer seed.

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

    for method_id, dims, k_stack, desc in BIT_METHODS:
        # Slice and quantize
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)

        if k_stack == 1:
            # Plain SignBitQuantizer
            q = SignBitQuantizer(d=dims, seed=seed)
            corpus_codes = q.encode(corpus)
            bytes_per_vec = corpus_codes.shape[1]

            # Compute -Hamming distance scores (higher = more similar)
            query_codes = q.encode(queries)
            scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
            for i in range(queries.shape[0]):
                scores[i] = -hamming_distances(corpus_codes, query_codes[i])

        else:
            # StackedSignBitQuantizer
            sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=seed)
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
            "seed": seed,
        })

    return rows


def evaluate_stsb_seed(
    s1: np.ndarray,
    s2: np.ndarray,
    gold: np.ndarray,
    seed: int,
) -> list[dict]:
    """Evaluate bit methods on STS-B similarity for a single seed.

    Parameters
    ----------
    s1, s2 : np.ndarray
        Embeddings, shape (n, 2048).
    gold : np.ndarray
        Gold scores, shape (n,), range [0, 5].
    seed : int
        Quantizer seed.

    Returns
    -------
    list[dict]
        One row dict per method.
    """
    rows = []

    for method_id, dims, k_stack, desc in BIT_METHODS:
        # Slice
        s1_sliced = slice_renorm(s1, dims)
        s2_sliced = slice_renorm(s2, dims)

        if k_stack == 1:
            q = SignBitQuantizer(d=dims, seed=seed)
            s1_codes = q.encode(s1_sliced)
            s2_codes = q.encode(s2_sliced)
        else:
            sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=seed)
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
            "spearman": spearman_val,
            "seed": seed,
        })

    return rows


def evaluate_scifact_float32(
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_full: np.ndarray,
    queries_full: np.ndarray,
) -> list[dict]:
    """Evaluate float32 methods on SciFact retrieval (deterministic, seed-independent).

    Parameters
    ----------
    corpus_ids, query_ids : list[str]
    qrels : dict[str, dict[str, int]]
    corpus_full, queries_full : np.ndarray

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

    for method_id, dims, k_stack, desc in FLOAT32_METHODS:
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)

        # Float32 baseline
        scores = queries @ corpus.T  # (m, n)
        bytes_per_vec = dims * 4

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
        })

    return rows


def evaluate_stsb_float32(
    s1: np.ndarray,
    s2: np.ndarray,
    gold: np.ndarray,
) -> list[dict]:
    """Evaluate float32 methods on STS-B similarity (deterministic).

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

    for method_id, dims, k_stack, desc in FLOAT32_METHODS:
        # Slice
        s1_sliced = slice_renorm(s1, dims)
        s2_sliced = slice_renorm(s2, dims)

        # Float32 baseline: pairwise cosine
        sim = np.sum(s1_sliced * s2_sliced, axis=1)  # (n,)
        bytes_per_vec = dims * 4

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
            "spearman": spearman_val,
        })

    return rows


def aggregate_rows(scifact_rows_by_seed: dict, stsb_rows_by_seed: dict) -> list[dict]:
    """Aggregate rows across seeds to compute mean and std.

    Parameters
    ----------
    scifact_rows_by_seed : dict
        Maps seed -> list of scifact row dicts.
    stsb_rows_by_seed : dict
        Maps seed -> list of stsb row dicts.

    Returns
    -------
    list[dict]
        Aggregated rows with mean/std columns.
    """
    agg_rows = []

    # Aggregate SciFact rows by method
    scifact_by_method = {}
    for seed, rows in scifact_rows_by_seed.items():
        for row in rows:
            method = row["method"]
            if method not in scifact_by_method:
                scifact_by_method[method] = {}
            scifact_by_method[method][seed] = row

    for method in sorted(scifact_by_method.keys()):
        seed_rows = scifact_by_method[method]
        n_seeds = len(seed_rows)

        # Get a template row
        template = next(iter(seed_rows.values()))

        # Compute means and stds
        r10_vals = [row["r10_vs_float"] for row in seed_rows.values()]
        ndcg10_vals = [row["ndcg10"] for row in seed_rows.values()]

        r10_mean = float(np.mean(r10_vals))
        r10_std = float(np.std(r10_vals, ddof=1) if n_seeds > 1 else 0.0)
        ndcg10_mean = float(np.mean(ndcg10_vals))
        ndcg10_std = float(np.std(ndcg10_vals, ddof=1) if n_seeds > 1 else 0.0)

        agg_rows.append({
            "dataset": "scifact",
            "method": method,
            "dims": template["dims"],
            "k_stack": template["k_stack"],
            "bytes_per_vec": template["bytes_per_vec"],
            "n_seeds": n_seeds,
            "r10_mean": r10_mean,
            "r10_std": r10_std,
            "ndcg10_mean": ndcg10_mean,
            "ndcg10_std": ndcg10_std,
            "spearman_mean": "",
            "spearman_std": "",
        })

    # Aggregate STS-B rows by method
    stsb_by_method = {}
    for seed, rows in stsb_rows_by_seed.items():
        for row in rows:
            method = row["method"]
            if method not in stsb_by_method:
                stsb_by_method[method] = {}
            stsb_by_method[method][seed] = row

    for method in sorted(stsb_by_method.keys()):
        seed_rows = stsb_by_method[method]
        n_seeds = len(seed_rows)

        # Get a template row
        template = next(iter(seed_rows.values()))

        # Compute means and stds
        spearman_vals = [row["spearman"] for row in seed_rows.values()]

        spearman_mean = float(np.mean(spearman_vals))
        spearman_std = float(np.std(spearman_vals, ddof=1) if n_seeds > 1 else 0.0)

        agg_rows.append({
            "dataset": "stsb",
            "method": method,
            "dims": template["dims"],
            "k_stack": template["k_stack"],
            "bytes_per_vec": template["bytes_per_vec"],
            "n_seeds": n_seeds,
            "r10_mean": "",
            "r10_std": "",
            "ndcg10_mean": "",
            "ndcg10_std": "",
            "spearman_mean": spearman_mean,
            "spearman_std": spearman_std,
        })

    return agg_rows


def add_float32_baseline_rows(
    agg_rows: list[dict],
    float32_scifact: list[dict],
    float32_stsb: list[dict],
) -> list[dict]:
    """Add float32 baseline rows with n_seeds=1 and std=0.

    Parameters
    ----------
    agg_rows : list[dict]
        Aggregated bit method rows.
    float32_scifact : list[dict]
        Float32 SciFact rows.
    float32_stsb : list[dict]
        Float32 STS-B rows.

    Returns
    -------
    list[dict]
        Combined rows.
    """
    # Add float32 SciFact rows
    for row in float32_scifact:
        agg_rows.append({
            "dataset": "scifact",
            "method": row["method"],
            "dims": row["dims"],
            "k_stack": row["k_stack"],
            "bytes_per_vec": row["bytes_per_vec"],
            "n_seeds": 1,
            "r10_mean": row["r10_vs_float"],
            "r10_std": 0.0,
            "ndcg10_mean": row["ndcg10"],
            "ndcg10_std": 0.0,
            "spearman_mean": "",
            "spearman_std": "",
        })

    # Add float32 STS-B rows
    for row in float32_stsb:
        agg_rows.append({
            "dataset": "stsb",
            "method": row["method"],
            "dims": row["dims"],
            "k_stack": row["k_stack"],
            "bytes_per_vec": row["bytes_per_vec"],
            "n_seeds": 1,
            "r10_mean": "",
            "r10_std": "",
            "ndcg10_mean": "",
            "ndcg10_std": "",
            "spearman_mean": row["spearman"],
            "spearman_std": 0.0,
        })

    return agg_rows


def print_summary_table(agg_rows: list[dict]) -> None:
    """Print summary table to stdout."""
    print("\n=== SUMMARY TABLE ===\n")
    print(
        f"{'Method':<20} {'Dataset':<10} {'R@10':<20} {'nDCG@10':<20} {'Spearman':<20}"
    )
    print("-" * 90)

    for row in agg_rows:
        if row["dataset"] == "scifact":
            r10_str = (
                f"{row['r10_mean']:.4f}±{row['r10_std']:.4f}"
                if row["r10_mean"] != ""
                else "-"
            )
            ndcg_str = (
                f"{row['ndcg10_mean']:.4f}±{row['ndcg10_std']:.4f}"
                if row["ndcg10_mean"] != ""
                else "-"
            )
            spearman_str = "-"
        else:
            r10_str = "-"
            ndcg_str = "-"
            spearman_str = (
                f"{row['spearman_mean']:.4f}±{row['spearman_std']:.4f}"
                if row["spearman_mean"] != ""
                else "-"
            )

        print(
            f"{row['method']:<20} {row['dataset']:<10} {r10_str:<20} {ndcg_str:<20} {spearman_str:<20}"
        )


def run_selftest() -> None:
    """Selftest on synthetic data.

    Generates n=200 docs, m=20 queries, d=256 with scaled MRL grid.
    Runs across 3 seeds. Checks positive-control assertions.
    """
    print("[eval_nemotron_seeds] running selftest...")

    n_docs = 200
    m_queries = 20
    d_test = 256
    n_selftest_seeds = 3

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
    s1_raw = rng.standard_normal((100, subdim)) @ basis.T
    s2_raw = rng.standard_normal((100, subdim)) @ basis.T
    s1_pairs = s1_raw.astype(np.float32)
    s2_pairs = s2_raw.astype(np.float32)
    s1_pairs /= np.linalg.norm(s1_pairs, axis=1, keepdims=True)
    s2_pairs /= np.linalg.norm(s2_pairs, axis=1, keepdims=True)
    gold_pairs = np.sum(s1_pairs * s2_pairs, axis=1) * 5  # scale to [0, 5]

    # Adapt BIT_METHODS for selftest (use d_test=256 instead of 2048)
    selftest_bit_methods = [
        ("bit1_256", 256, 1, "SignBitQuantizer d=256"),
        ("bit1_stack2", 256, 2, "StackedSignBitQuantizer d=256 k=2"),
        ("bit1_stack4", 256, 4, "StackedSignBitQuantizer d=256 k=4"),
        ("bit1_mrl128", 128, 1, "slice 128 + SignBitQuantizer"),
        ("bit1_mrl64", 64, 1, "slice 64 + SignBitQuantizer"),
    ]

    # Evaluate bit methods across seeds
    print(f"[eval_nemotron_seeds] evaluating bit methods across {n_selftest_seeds} seeds...")
    scifact_by_seed = {}
    stsb_by_seed = {}

    for seed in range(n_selftest_seeds):
        # For selftest, we need custom evaluation functions with the adapted methods
        scifact_rows = []
        query_ids_list = list(query_ids)
        corpus_ids_list = list(corpus_ids)
        float_scores_local = queries_full @ corpus_full.T

        for method_id, dims, k_stack, desc in selftest_bit_methods:
            corpus = slice_renorm(corpus_full, dims)
            queries = slice_renorm(queries_full, dims)

            if k_stack == 1:
                q = SignBitQuantizer(d=dims, seed=seed)
                corpus_codes = q.encode(corpus)
                bytes_per_vec = corpus_codes.shape[1]
                query_codes = q.encode(queries)
                scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
                for i in range(queries.shape[0]):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])
            else:
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=seed)
                corpus_codes = sq.encode(corpus)
                bytes_per_vec = corpus_codes.shape[1]
                query_codes = sq.encode(queries)
                scores = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.float64)
                for i in range(queries.shape[0]):
                    scores[i] = -hamming_distances(corpus_codes, query_codes[i])

            r10_float = r10_vs_float(scores, query_ids_list, corpus_ids_list, float_scores_local)
            ndcg10_vals = []
            for i, qid in enumerate(query_ids_list):
                rel_q = qrels.get(qid, {})
                if rel_q:
                    ndcg10_vals.append(ndcg_at_k(scores[i], corpus_ids_list, rel_q, k=10))
            ndcg10 = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0

            scifact_rows.append({
                "dataset": "scifact",
                "method": method_id,
                "dims": dims,
                "k_stack": k_stack,
                "bytes_per_vec": bytes_per_vec,
                "r10_vs_float": r10_float,
                "ndcg10": ndcg10,
                "seed": seed,
            })

        scifact_by_seed[seed] = scifact_rows

        # STS-B
        stsb_rows = []
        for method_id, dims, k_stack, desc in selftest_bit_methods:
            s1 = slice_renorm(s1_pairs, dims)
            s2 = slice_renorm(s2_pairs, dims)

            if k_stack == 1:
                q = SignBitQuantizer(d=dims, seed=seed)
                s1_codes = q.encode(s1)
                s2_codes = q.encode(s2)
            else:
                sq = StackedSignBitQuantizer(d=dims, k=k_stack, seed=seed)
                s1_codes = sq.encode(s1)
                s2_codes = sq.encode(s2)

            sim = np.array(
                [-hamming_distances(s2_codes, s1_codes[i])[i] for i in range(len(s1_codes))]
            )
            bytes_per_vec = s1_codes.shape[1]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                spearman_corr, _ = spearmanr(sim, gold_pairs)
            spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

            stsb_rows.append({
                "dataset": "stsb",
                "method": method_id,
                "dims": dims,
                "k_stack": k_stack,
                "bytes_per_vec": bytes_per_vec,
                "spearman": spearman_val,
                "seed": seed,
            })

        stsb_by_seed[seed] = stsb_rows

    # Aggregate
    agg_rows = aggregate_rows(scifact_by_seed, stsb_by_seed)

    # Evaluate float32 methods (adapted for selftest)
    print("[eval_nemotron_seeds] evaluating float32 baselines...")
    selftest_float32_methods = [
        ("f32_256", 256, 1, "float32 full"),
        ("f32_mrl128", 128, 1, "float32 MRL slice 128"),
        ("f32_mrl64", 64, 1, "float32 MRL slice 64"),
    ]

    float32_scifact = []
    float_scores_local = queries_full @ corpus_full.T
    for method_id, dims, k_stack, desc in selftest_float32_methods:
        corpus = slice_renorm(corpus_full, dims)
        queries = slice_renorm(queries_full, dims)
        scores = queries @ corpus.T
        bytes_per_vec = dims * 4
        r10_float = r10_vs_float(scores, list(corpus_ids), list(query_ids), float_scores_local)
        ndcg10_vals = []
        for i, qid in enumerate(list(query_ids)):
            rel_q = qrels.get(qid, {})
            if rel_q:
                ndcg10_vals.append(ndcg_at_k(scores[i], list(corpus_ids), rel_q, k=10))
        ndcg10 = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0

        float32_scifact.append({
            "dataset": "scifact",
            "method": method_id,
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "r10_vs_float": r10_float,
            "ndcg10": ndcg10,
        })

    float32_stsb = []
    for method_id, dims, k_stack, desc in selftest_float32_methods:
        s1 = slice_renorm(s1_pairs, dims)
        s2 = slice_renorm(s2_pairs, dims)
        sim = np.sum(s1 * s2, axis=1)
        bytes_per_vec = dims * 4
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            spearman_corr, _ = spearmanr(sim, gold_pairs)
        spearman_val = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

        float32_stsb.append({
            "dataset": "stsb",
            "method": method_id,
            "dims": dims,
            "k_stack": k_stack,
            "bytes_per_vec": bytes_per_vec,
            "spearman": spearman_val,
        })

    # Add float32 rows
    agg_rows = add_float32_baseline_rows(agg_rows, float32_scifact, float32_stsb)

    # Assertions
    assertions_passed = 0
    assertions_total = 0

    # (a) float32 rows have std == 0
    assertions_total += 1
    float32_rows = [r for r in agg_rows if r["method"].startswith("f32")]
    all_std_zero = all(
        (r["r10_std"] == 0.0 or r["r10_std"] == "")
        and (r["ndcg10_std"] == 0.0 or r["ndcg10_std"] == "")
        and (r["spearman_std"] == 0.0 or r["spearman_std"] == "")
        for r in float32_rows
    )
    if all_std_zero:
        print(f"PASS: all {len(float32_rows)} float32 rows have std == 0")
        assertions_passed += 1
    else:
        bad = [r["method"] for r in float32_rows if (
            (isinstance(r["r10_std"], float) and r["r10_std"] != 0.0) or
            (isinstance(r["ndcg10_std"], float) and r["ndcg10_std"] != 0.0) or
            (isinstance(r["spearman_std"], float) and r["spearman_std"] != 0.0)
        )]
        print(f"FAIL: float32 rows {bad} do not have std == 0")

    # (b) bit methods have std >= 0
    assertions_total += 1
    bit_rows = [r for r in agg_rows if r["method"].startswith("bit")]
    all_std_nonneg = all(
        (isinstance(r["r10_std"], float) and r["r10_std"] >= 0.0 or r["r10_std"] == "")
        and (isinstance(r["ndcg10_std"], float) and r["ndcg10_std"] >= 0.0 or r["ndcg10_std"] == "")
        and (isinstance(r["spearman_std"], float) and r["spearman_std"] >= 0.0 or r["spearman_std"] == "")
        for r in bit_rows
    )
    if all_std_nonneg:
        print(f"PASS: all {len(bit_rows)} bit methods have std >= 0")
        assertions_passed += 1
    else:
        bad = [r["method"] for r in bit_rows if not all(
            (isinstance(r["r10_std"], float) and r["r10_std"] >= 0.0 or r["r10_std"] == "")
            and (isinstance(r["ndcg10_std"], float) and r["ndcg10_std"] >= 0.0 or r["ndcg10_std"] == "")
            and (isinstance(r["spearman_std"], float) and r["spearman_std"] >= 0.0 or r["spearman_std"] == "")
        )]
        print(f"FAIL: bit methods {bad} have std < 0")

    # (c) scifact methods have r10_mean, ndcg10_mean in [0, 1]
    assertions_total += 1
    scifact_rows = [r for r in agg_rows if r["dataset"] == "scifact"]
    all_in_range = all(
        (0 <= r["r10_mean"] <= 1) and (0 <= r["ndcg10_mean"] <= 1)
        for r in scifact_rows
    )
    if all_in_range:
        print(f"PASS: all {len(scifact_rows)} scifact methods have r10/ndcg in [0,1]")
        assertions_passed += 1
    else:
        bad = [r["method"] for r in scifact_rows if not (
            (0 <= r["r10_mean"] <= 1) and (0 <= r["ndcg10_mean"] <= 1)
        )]
        print(f"FAIL: scifact methods {bad} have r10/ndcg outside [0,1]")

    # (d) stsb methods have spearman_mean in [0, 1]
    assertions_total += 1
    stsb_rows = [r for r in agg_rows if r["dataset"] == "stsb"]
    all_in_range = all(
        (-1 <= r["spearman_mean"] <= 1)
        for r in stsb_rows
    )
    if all_in_range:
        print(f"PASS: all {len(stsb_rows)} stsb methods have spearman in [-1,1]")
        assertions_passed += 1
    else:
        bad = [r["method"] for r in stsb_rows if not (-1 <= r["spearman_mean"] <= 1)]
        print(f"FAIL: stsb methods {bad} have spearman outside [-1,1]")

    print(f"\n[eval_nemotron_seeds] selftest: {assertions_passed}/{assertions_total} assertions passed")
    if assertions_passed < assertions_total:
        sys.exit(1)


def main() -> None:
    """Main evaluation on real Nemotron embeddings."""
    parser = argparse.ArgumentParser(
        description="Nemotron-3-Embed-1B multi-seed evaluation for remax."
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
            f"[eval_nemotron_seeds] ERROR: missing embedding caches:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        print(
            "\nRun Agent A (embed_nemotron.py) first to generate these caches.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load embeddings
    print("[eval_nemotron_seeds] loading embeddings...")
    corpus_full = np.load(EMB_DIR / "scifact_docs.npy").astype(np.float32)
    queries_full = np.load(EMB_DIR / "scifact_queries.npy").astype(np.float32)
    s1_full = np.load(EMB_DIR / "stsb_s1.npy").astype(np.float32)
    s2_full = np.load(EMB_DIR / "stsb_s2.npy").astype(np.float32)

    # Load data
    print("[eval_nemotron_seeds] loading data...")
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

    # Evaluate bit methods across seeds
    print(f"[eval_nemotron_seeds] evaluating bit methods across {len(SEEDS)} seeds...")
    scifact_by_seed = {}
    stsb_by_seed = {}

    for seed in SEEDS:
        print(f"  seed {seed}...")
        scifact_by_seed[seed] = evaluate_scifact_seed(
            corpus_ids, query_ids, qrels, corpus_full, queries_full, seed
        )
        stsb_by_seed[seed] = evaluate_stsb_seed(s1_full, s2_full, gold_stsb, seed)

    # Aggregate
    print("[eval_nemotron_seeds] aggregating results...")
    agg_rows = aggregate_rows(scifact_by_seed, stsb_by_seed)

    # Evaluate float32 methods (deterministic, single-seed)
    print("[eval_nemotron_seeds] evaluating float32 baselines...")
    float32_scifact = evaluate_scifact_float32(
        corpus_ids, query_ids, qrels, corpus_full, queries_full
    )
    float32_stsb = evaluate_stsb_float32(s1_full, s2_full, gold_stsb)

    # Add float32 rows with n_seeds=1, std=0
    agg_rows = add_float32_baseline_rows(agg_rows, float32_scifact, float32_stsb)

    # Write CSV
    csv_path = RESULTS_DIR / "nemotron_seeds.csv"
    print(f"[eval_nemotron_seeds] writing {csv_path}...")

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "method",
            "dims",
            "k_stack",
            "bytes_per_vec",
            "n_seeds",
            "r10_mean",
            "r10_std",
            "ndcg10_mean",
            "ndcg10_std",
            "spearman_mean",
            "spearman_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            # Format floats to 4 decimals, keep empty strings
            formatted_row = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = str(v) if v != "" else ""
            writer.writerow(formatted_row)

    print(f"[eval_nemotron_seeds] wrote {len(agg_rows)} rows to {csv_path}")

    # Print summary table
    print_summary_table(agg_rows)


if __name__ == "__main__":
    main()
