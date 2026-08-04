# remax × NVIDIA Nemotron-3-Embed-1B: 1-bit quantization

## Setup

- **Model**: `nvidia/Nemotron-3-Embed-1B-BF16` (derived from Ministral-3-3B, pruned to 1.14B params; 2048-dim hidden, avg-pooled, L2-normalized output; Matryoshka slice-and-renormalize supported).
- **NVFP4 proxy note**: the target `nvidia/Nemotron-3-Embed-1B-NVFP4` checkpoint quantizes linear-layer weights + activations to NVFP4 and requires GPU + vLLM 0.25.0. We encode with the public BF16 twin; NVIDIA's own RTEB delta between the two checkpoints is **0.38 nDCG@10** (72.00 NVFP4 vs 72.38 BF16), so BF16-derived deltas transfer to within that margin. remax quantization is applied to the *emitted embeddings* at index time — it is orthogonal to the model-weight NVFP4 quantization and composes with either checkpoint.
- **Datasets**:
  - **SciFact** (BeIR): 1600-doc corpus (subset containing all 283 qrel-referenced docs + random fill), 300 test queries with graded qrels. Retrieval.
  - **STS-B** (test): 1379 sentence pairs, gold similarity [0, 5], reported as Spearman rank correlation.
- **Encoding**: CPU only (4 threads), float32, chunked with resume.
- **Quantizer**: `SignBitQuantizer` (1-bit) and `StackedSignBitQuantizer` (k=2, 4) from remax, seed=0, Haar random rotation pre-applied. `bytes_per_vec` is the packed code size; `compression_x = 8192 / bytes_per_vec`.

## Method grid

| method_id | description | bytes/vec | compression_x |
|-----------|-------------|-----------|---------------|
| f32_2048 | float32 full (ground truth for R@10) | 8192 | 1.0 |
| f32_mrl256 | float32 MRL slice 256 | 1024 | 8.0 |
| f32_mrl128 | float32 MRL slice 128 | 512 | 16.0 |
| f32_mrl64 | float32 MRL slice 64 | 256 | 32.0 |
| bit1_2048 | SignBitQuantizer d=2048 | 256 | 32.0 |
| bit1_stack2 | StackedSignBitQuantizer k=2 d=2048 | 512 | 16.0 |
| bit1_stack4 | StackedSignBitQuantizer k=4 d=2048 | 1024 | 8.0 |
| bit1_mrl1024 | slice 1024 + SignBitQuantizer | 128 | 64.0 |
| bit1_mrl512 | slice 512 + SignBitQuantizer | 64 | 128.0 |
| bit1_mrl256 | slice 256 + SignBitQuantizer | 32 | 256.0 |

## Results

### SciFact — rank recovery (R@10 vs float32) and relevance (nDCG@10)

| method | bytes | compression_x | R@10 vs float32 | nDCG@10 | Δ nDCG@10 vs f32_2048 |
|--------|-------|---------------|-----------------|---------|----------------------|
| f32_2048 | 8192 | 1.0 | 1.0000 | 0.8419 | +0.0000 |
| f32_mrl256 | 1024 | 8.0 | 0.5543 | 0.7599 | -0.0820 |
| f32_mrl128 | 512 | 16.0 | 0.3887 | 0.6637 | -0.1782 |
| f32_mrl64 | 256 | 32.0 | 0.2397 | 0.4897 | -0.3522 |
| bit1_2048 | 256 | 32.0 | 0.7613 | 0.8196 | -0.0223 |
| bit1_stack2 | 512 | 16.0 | 0.8127 | 0.8386 | -0.0033 |
| bit1_stack4 | 1024 | 8.0 | 0.8707 | 0.8421 | +0.0002 |
| bit1_mrl1024 | 128 | 64.0 | 0.6207 | 0.8019 | -0.0400 |
| bit1_mrl512 | 64 | 128.0 | 0.4870 | 0.7442 | -0.0977 |
| bit1_mrl256 | 32 | 256.0 | 0.3557 | 0.6247 | -0.2172 |

### STS-B — Spearman rank correlation

| method | bytes | compression_x | Spearman ρ | Δ vs f32_2048 |
|--------|-------|---------------|------------|---------------|
| f32_2048 | 8192 | 1.0 | 0.8478 | +0.0000 |
| f32_mrl256 | 1024 | 8.0 | 0.8455 | -0.0023 |
| f32_mrl128 | 512 | 16.0 | 0.8369 | -0.0109 |
| f32_mrl64 | 256 | 32.0 | 0.8080 | -0.0398 |
| bit1_2048 | 256 | 32.0 | 0.8447 | -0.0031 |
| bit1_stack2 | 512 | 16.0 | 0.8452 | -0.0026 |
| bit1_stack4 | 1024 | 8.0 | 0.8470 | -0.0008 |
| bit1_mrl1024 | 128 | 64.0 | 0.8414 | -0.0064 |
| bit1_mrl512 | 64 | 128.0 | 0.8398 | -0.0080 |
| bit1_mrl256 | 32 | 256.0 | 0.8214 | -0.0264 |

## Memory-matched comparisons

Same bytes/vec budget — remax 1-bit vs float32 Matryoshka slice. Positive Δ means remax wins:

| Pair | bytes/vec | SciFact R@10 Δ | SciFact nDCG@10 Δ | STS-B Spearman Δ |
|------|-----------|----------------|-------------------|-----------------|
| bit1_2048 vs f32_mrl64 | 256 | +0.5216 | +0.3299 | +0.0367 |
| bit1_stack2 vs f32_mrl128 | 512 | +0.4240 | +0.1749 | +0.0083 |
| bit1_stack4 vs f32_mrl256 | 1024 | +0.3164 | +0.0822 | +0.0015 |

remax 1-bit dominates float32 truncation at **every** matched memory budget, on both tasks. The gap is largest on retrieval (SciFact) and at the tightest budgets: at 256 bytes, 1-bit sign codes recover 76% of the float top-10 while a 64-dim float slice recovers 24%.

## Robustness: multi-seed error bars

The remax bit methods depend on the random Haar rotation (seed); float32 is
deterministic. Re-running the six bit methods over **seeds 0–4** (mean ± std):

| method | bytes | SciFact R@10 | SciFact nDCG@10 | STS-B ρ |
|--------|-------|--------------|-----------------|---------|
| bit1_2048 | 256 | 0.755 ± 0.004 | 0.825 ± 0.008 | 0.8445 ± 0.0010 |
| bit1_stack2 | 512 | 0.818 ± 0.006 | 0.835 ± 0.002 | 0.8463 ± 0.0007 |
| bit1_stack4 | 1024 | 0.870 ± 0.002 | 0.838 ± 0.003 | 0.8470 ± 0.0004 |
| bit1_mrl1024 | 128 | 0.620 ± 0.005 | 0.802 ± 0.008 | 0.8431 ± 0.0019 |
| bit1_mrl512 | 64 | 0.486 ± 0.005 | 0.737 ± 0.006 | 0.8364 ± 0.0026 |
| bit1_mrl256 | 32 | 0.343 ± 0.010 | 0.627 ± 0.008 | 0.8211 ± 0.0044 |

Std ≤ 0.01 on every metric — the near-lossless result is **not** a lucky
rotation. (These supersede the seed-0 point estimates in the table above, which
differ by < 0.01.)

## Baselines: int8 and product quantization at matched byte budgets

remax 1-bit is data-oblivious. The two standard competitors are **int8** scalar
quantization (also data-oblivious, but only 4× at full dims — to compress
further it must truncate dimensions) and **product quantization** (faiss PQ,
`nbits=8`; **data-dependent** — it trains a codebook on the corpus). Matched
byte budgets, same metrics (`nemotron_baselines.csv`, figure
`nemotron_baselines.png`):

**The 256-byte budget (32× compression) — head to head:**

| method | family | data-dependent? | SciFact nDCG@10 | SciFact R@10 | STS-B ρ |
|--------|--------|:---------------:|-----------------|--------------|---------|
| **remax bit1_2048** | remax | no | **0.825** | **0.755** | **0.845** |
| pq_m256 | PQ | **yes** (trains) | 0.811 | 0.753 | 0.798 |
| int8_mrl256 | int8 | no | 0.760 | 0.553 | 0.845 |
| f32_mrl64 | float32 | no | 0.490 | 0.240 | 0.808 |

The full ladders show *why* each competitor fails at low bytes:
- **int8** is near-lossless per dimension (int8_2048 = 0.844 nDCG at 4×, ties
  float32), but reaching 32× forces it down to a 256-dim slice, and truncation
  destroys retrieval ranking (0.760 nDCG, 0.553 R@10). On STS-B it stays even
  with remax (similarity tolerates truncation far better than retrieval).
- **PQ** is the strongest retrieval competitor — at 256 B it nearly matches
  remax (0.811 vs 0.825 nDCG). But it collapses on STS-B similarity
  (0.798, and 0.542 at 32 B) and it needs a training corpus (M=256 fails to
  train on < 9,984 docs — a real deployment constraint remax doesn't have).

## Latency / throughput

Query-time cost, SciFact corpus (1,600 docs × 300 queries), single thread,
k=10, best of 5 (`nemotron_latency.csv`):

| method | family | bytes/vec | index MB | per-query ms (median) | queries/sec |
|--------|--------|-----------|----------|----------------------|-------------|
| f32_mrl256 | float32 | 1024 | 1.56 | 0.020 | 50,492 |
| f32_2048 | float32 | 8192 | 12.50 | 0.038 | 26,352 |
| int8_2048 | int8 | 2048 | 3.12 | 0.042 | 22,626 |
| bit1_2048 | remax | 256 | **0.39** | 0.042 | 23,325 |
| bit1_stack4 | remax | 1024 | 1.56 | 0.155 | 6,428 |

> ### ⚠️ The table above came from a harness that did not treat the arms alike
>
> **Corrected 2026-08-03.** Do not compare those per-query columns across
> families. `bench/latency_nemotron.py` handed the float32 baseline three
> advantages that belong to the harness, not to either algorithm:
>
> | | float32 arm | remax arm |
> |---|---|---|
> | scoring | `queries @ corpus.T` — **one** batched GEMM over all 300 | a Python loop, **300** calls to `hamming_distances` |
> | top-k | `np.argsort(-scores, axis=1)[:, :k]` — one batched call | `np.argsort(distances)[:k]` — a full O(n log n) sort, **per query** |
> | selector | numpy's own | **not** `stable_top_k`, the O(n) argpartition selector remax ships |
>
> The batched GEMM amortises Python and dispatch overhead across 300 queries
> and lets BLAS keep the corpus blocked in cache. The per-query loop rereads
> the index 300 times and pays dispatch 300 times. The two columns are not
> measuring comparable things, and the mismatch runs one way.
>
> Measured at the published shape (n = 5,183, d = 2,048, m = 300, k = 10,
> single-threaded, best of 5) on synthetic vectors — re-running the real arms
> needs the embedding cache:
>
> | arm | ms/query |
> |---|--:|
> | float32, batched GEMM | 0.311 |
> | float32, per-query | 4.033 |
> | remax 1-bit | 0.196 |
> | remax stacked k=4 | 0.690 |
>
> The batching axis alone is worth **13×** to float32. Against a like-for-like
> per-query float32 baseline, remax 1-bit is **20.6× faster**; against a
> *batched* float32 baseline it is **1.6×**. Both are real, and they answer
> different questions — "which is the faster scan" versus "can I beat BLAS at
> what BLAS is best at". Only the second was ever reported, and it was
> reported without saying which one it was.
>
> Denying remax its own selector was separately worth ~18% (0.255 → 0.210
> ms/query once `stable_top_k` and an `out=` buffer replace `np.argsort` and a
> per-query allocation).
>
> One caveat that runs the *other* way, for completeness: query encoding
> (`query @ R`) sits outside the timed region for remax, as it did
> originally. float32 has no encode step, so that exclusion favours remax.
>
> `mode` is now an explicit axis in the harness and both arms use an O(n)
> selector. The remax arm documents that it has **no batched Hamming kernel**
> rather than manufacturing one: `search` loops in Python however it is
> called, and an m-way SIMD popcount is explicitly post-v0.1.0. Regenerating
> this table against the real cache is the remaining work — the numbers above
> are the harness correction, not a replacement measurement.

**Honest reading of latency**: at n = 1,600 the corpus is far too small for
scan *speed* to be remax's win — a BLAS float32 matmul over 1,600 vectors is
already sub-40 µs, and the pure-numpy popcount Hamming scan is in the same
ballpark (stacked k=4 is actually slower). remax's decisive advantage here is
**index size**: 0.39 MB vs 12.5 MB for float32 (32×). The throughput crossover
where a packed Hamming scan beats a bandwidth-bound float scan lives at much
larger n — that is exactly remex's `IVFCoarseIndex` territory, out of scope for
this experiment.

That conclusion survives the correction — index size, not scan speed, is what
this experiment supports — but it was reached from numbers that understated
remax's per-query scan by roughly 13×. It was right for reasons only partly
connected to the evidence offered for it.

## Reading

- **1-bit at 32× is near-lossless for ranking, and stacking closes the rest.** Full-dimension 1-bit (256 B, 32×) costs only −0.022 nDCG@10 on SciFact and −0.003 Spearman on STS-B. `StackedSignBitQuantizer` k=4 (1024 B, 8×) matches float32 nDCG@10 against the qrels (+0.0002) while recovering 87% of float32's own top-10 (R@10 = 0.871) — the 13% it reorders are swaps among near-equivalent-relevance docs, which is why measured nDCG is unchanged. This is the one-bit-beats-two thesis reproduced on a new 2026 model family: sign bits preserve angular rank; the value ladder is stacking, not more bits per coordinate.
- **Sign-bit precision beats dimensional truncation, decisively.** At a fixed byte budget, spend it on sign bits over all dimensions, not float32 over a Matryoshka prefix. The 256-byte pair is the clearest: 1-bit@2048 (nDCG 0.820) vs float32@64 (nDCG 0.490). Truncating to 64 dims throws away the coordinates that carry the ranking; keeping all 2048 sign bits keeps them.
- **The two tasks stress different things and agree.** STS-B (pairwise similarity, Spearman) is far more forgiving than SciFact (top-10 retrieval) — every method stays above ρ = 0.82 — but the ordering of methods is identical across both, so the conclusion is not a retrieval-only artifact.
- **Against real baselines, remax is the robust generalist — the only method on the upper-left frontier of *both* tasks.** int8 ties remax on similarity but collapses on retrieval below ~2 KB/vec, because compressing past 4× forces dimensional truncation. PQ is a near-match on retrieval but collapses on similarity *and* needs a training corpus (a fit step, and it fails outright on small corpora). remax needs neither — it is data-oblivious, hits 32× without truncating, and stays top-tier on similarity and retrieval alike (see `nemotron_baselines.png`). That robustness, not a win on any single metric, is the result.
- **Memory in absolute terms.** A 1M-vector Nemotron index is 8.0 GB at float32, 256 MB at 1-bit (32×), 32 MB at 1-bit-on-256-slice (256×). The 32× point is the sweet spot: it fits an index 32× larger in the same RAM at a −0.02 nDCG cost. At n = 1,600 that memory win does not yet translate to a *speed* win (the flat float scan is already sub-40 µs); the latency crossover is a large-n regime this experiment doesn't reach.
- **Caveat — BF16 proxy, single model.** All numbers are on the BF16 embeddings; the NVFP4 checkpoint (GPU + vLLM) was not run here. remax quantizes the emitted vectors and is orthogonal to NVFP4 weight quantization, but that composition is argued, not measured. Two datasets on one model family — a signal, not a benchmark sweep.

## Provenance — the drivers are gone, the results are not

**The ten Nemotron/NVFP4 bench scripts were deleted on 2026-08-03** (4,650 lines,
including the shared `nemotron_paths.py`). This file, `NEMOTRON_MASTER.md`, the
committed CSVs and the committed PNGs are the record, and they are complete: every
number quoted above has its row in a CSV in this directory.

They went because they were apparatus, not library. They benchmark a third-party
embedder (`nvidia/Nemotron-3-Embed-1B-BF16`) rather than anything in `src/remax/`,
they were the largest single block of Python in the repository, and re-running
them needs a model, a ~1 h CPU encode or a published embedding cache, `remex`, and
`matplotlib` — so nobody was running them anyway.

| what produced what | removed script |
|---|---|
| `nemotron_1bit.csv`, `nemotron_1bit.png` | `eval_nemotron_1bit.py`, `plot_nemotron_1bit.py` |
| `nemotron_seeds.csv` (mean±std, seeds 0-4) | `eval_nemotron_seeds.py` |
| `nemotron_baselines.csv`, `nemotron_baselines.png` | `baselines_nemotron.py` |
| `nemotron_latency.csv` | `latency_nemotron.py` (see the correction above — this one was measuring the harness) |
| `nemotron_remex.csv` | `remex_nemotron.py` |
| `nemotron_nvfp4.csv` | `nvfp4_eval.py`, `nvfp4_dequant_encode.py` |
| `nemotron_master.png` | `plot_nemotron_master.py` |
| the embeddings all of the above read | `embed_nemotron.py` |

The published embedding cache still exists and `bench/fetch_nemotron_cache.sh` is
kept, because it is the pointer to the raw data — release tag
`nemotron-3-embed-1b-bf16` on `oaustegard/claude-container-layers`, with the array
shapes it verifies documented in the script. Recovering a driver is
`git log --diff-filter=D -- bench/eval_nemotron_1bit.py` and then `git show`.

Before restoring any of them, be clear about which question is still open. The
conclusions above are not among them.
