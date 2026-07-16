# remax × NVIDIA Nemotron-3-Embed-1B: 1-bit quantization

## Setup

- **Model**: `nvidia/Nemotron-3-Embed-1B` (1.1B params, 2048-dim, L2-normalized, Matryoshka slicing supported).
- **Encoding note**: The NVFP4 (NF4) checkpoint requires GPU + vLLM. We encode with the public BF16 variant; NVIDIA's own RTEB benchmark delta between them is 0.38 nDCG@10 points on average.
- **Datasets**:
  - **SciFact**: 1600-doc corpus (subset with all 283 qrel docs), 300 test queries. Binary relevance.
  - **STS-B**: Test set, 1379 pairs. Gold similarity scores [0, 5], reported as Spearman rank correlation.
- **Encoding**: CPU only. Embeddings cached at session start.
- **Quantizer**: SignBitQuantizer (1-bit) and StackedSignBitQuantizer (k=2,4) from remax v0.1.0. Seed=0. Haar random rotation pre-applied.

## Method grid

| method_id | description | bytes/vec | compression_x |
|-----------|-------------|-----------|---------------|
| f32_2048 | float32 full (ground truth) | 8192 | 1.0 |
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

### SciFact — Rank recovery (R@10 vs float32) and relevance (nDCG@10)

| method | bytes | compression_x | R@10 vs float32 | nDCG@10 | Δ nDCG@10 vs f32_2048 |
|--------|-------|---------------|-----------------|---------|----------------------|
| f32_2048 | 8192 | 1.0 | 1.0 | TODO | 0.0 |
| f32_mrl256 | 1024 | 8.0 | TODO | TODO | TODO |
| f32_mrl128 | 512 | 16.0 | TODO | TODO | TODO |
| f32_mrl64 | 256 | 32.0 | TODO | TODO | TODO |
| bit1_2048 | 256 | 32.0 | TODO | TODO | TODO |
| bit1_stack2 | 512 | 16.0 | TODO | TODO | TODO |
| bit1_stack4 | 1024 | 8.0 | TODO | TODO | TODO |
| bit1_mrl1024 | 128 | 64.0 | TODO | TODO | TODO |
| bit1_mrl512 | 64 | 128.0 | TODO | TODO | TODO |
| bit1_mrl256 | 32 | 256.0 | TODO | TODO | TODO |

### STS-B — Spearman rank correlation

| method | bytes | compression_x | Spearman ρ | Δ vs f32_2048 |
|--------|-------|---------------|------------|---------------|
| f32_2048 | 8192 | 1.0 | TODO | 0.0 |
| f32_mrl256 | 1024 | 8.0 | TODO | TODO |
| f32_mrl128 | 512 | 16.0 | TODO | TODO |
| f32_mrl64 | 256 | 32.0 | TODO | TODO |
| bit1_2048 | 256 | 32.0 | TODO | TODO |
| bit1_stack2 | 512 | 16.0 | TODO | TODO |
| bit1_stack4 | 1024 | 8.0 | TODO | TODO |
| bit1_mrl1024 | 128 | 64.0 | TODO | TODO |
| bit1_mrl512 | 64 | 128.0 | TODO | TODO |
| bit1_mrl256 | 32 | 256.0 | TODO | TODO |

## Memory-matched comparisons

Three dimensionally equivalent pairs (same bytes/vec budget) — remax 1-bit vs float32 MRL:

| Pair | bytes/vec | ReMax method | float32 baseline | SciFact R@10 Δ | SciFact nDCG@10 Δ | STS-B Spearman Δ |
|------|-----------|--------------|------------------|----------------|-------------------|-----------------|
| 1 | 256 | bit1_2048 | f32_mrl64 | TODO | TODO | TODO |
| 2 | 512 | bit1_stack2 | f32_mrl128 | TODO | TODO | TODO |
| 3 | 1024 | bit1_stack4 | f32_mrl256 | TODO | TODO | TODO |

**Observation** (placeholder): TODO. Discuss which memory budget favors which approach and implications for deployment.

## Reading

- **TODO**: 1-bit vs stacked precision tradeoff — when to use k=1 vs k=2/4 for the same memory budget.
- **TODO**: Matryoshka (MRL) dimensionality vs bit precision — which dimension-reduction lever is more effective?
- **TODO**: Cross-dataset robustness — does the pattern generalize (SciFact retrieval vs STS-B similarity)?
