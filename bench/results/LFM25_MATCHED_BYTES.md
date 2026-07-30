# LFM2.5-Embedding-350M — codec bake-off at matched byte budgets (BEIR SciFact)

Chart: `lfm25_pareto.png`. Source: `lfm25_scifact.json` (written by `bench/eval_lfm25.py`).

Uncompressed reference: **fp32 d=1024** at 4096 B/vec, nDCG@10 **0.7122**. Every percentage below is relative to it.

Geometry called it **isotropic** (post-rotation sigma ratio 0.937) — predicted: remex dominates at every matched byte budget; 4-bit near-lossless

## Matched bytes — best codec per family at each budget

Cell = best nDCG@10 that family reaches at that budget, with the codec that got it. **Bold** = winner of the budget.

| bytes/vec | fp32 | int8 | remex | remax-unc | remax |
|---|:-:|:-:|:-:|:-:|:-:|
| 4096 | **0.712 (fp32 d=1024)** | — | — | — | — |
| 2048 | **0.704 (d=512)** | — | — | — | — |
| 1024 | 0.657 (d=256) | **0.712 (int8 d=1024)** | 0.712 (8bit) | — | 0.709 (k=8) |
| 512 | 0.609 (d=128) | — | **0.711 (4bit)** | — | 0.708 (k=4) |
| 384 | — | — | **0.710 (3bit)** | — | — |
| 256 | 0.509 (d=64) | — | **0.704 (2bit)** | — | 0.699 (k=2) |
| 128 | — | — | **0.702 (1bit)** | 0.681 (1bit uncentered) | 0.677 (1bit) |
| 64 | — | — | — | — | **0.650 (1bit d=512)** |
| 32 | — | — | — | — | **0.572 (1bit d=256)** |
| 16 | — | — | — | — | **0.421 (1bit d=128)** |
| 8 | — | — | — | — | **0.204 (1bit d=64)** |

## Every codec at every budget

### 4096 bytes/vector — 1.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| fp32 d=1024 | fp32 | 0.7122 | 100.0% | 0.8336 | 0.9617 | 1.0000 |

### 2048 bytes/vector — 2.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| fp32 d=512 | fp32 | 0.7042 | 98.9% | 0.8386 | 0.9600 | 0.7880 |

### 1024 bytes/vector — 4.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| **int8 d=1024** | int8 | **0.7122** | 100.0% | 0.8336 | 0.9617 | 0.9960 |
| remex 8bit d=1024 | remex | 0.7122 | 100.0% | 0.8336 | 0.9617 | 0.9960 |
| remax k=8 d=1024 | remax | 0.7087 | 99.5% | 0.8369 | 0.9567 | 0.8190 |
| fp32 d=256 | fp32 | 0.6572 | 92.3% | 0.7861 | 0.9450 | 0.6380 |

### 512 bytes/vector — 8.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| **remex 4bit d=1024** | remex | **0.7110** | 99.8% | 0.8336 | 0.9583 | 0.9617 |
| remax k=4 d=1024 | remax | 0.7078 | 99.4% | 0.8391 | 0.9633 | 0.7880 |
| fp32 d=128 | fp32 | 0.6085 | 85.4% | 0.7353 | 0.8993 | 0.4797 |

### 384 bytes/vector — 10.7x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| remex 3bit d=1024 | remex | 0.7097 | 99.6% | 0.8358 | 0.9617 | 0.9360 |

### 256 bytes/vector — 16.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| **remex 2bit d=1024** | remex | **0.7042** | 98.9% | 0.8324 | 0.9550 | 0.8830 |
| remax k=2 d=1024 | remax | 0.6989 | 98.1% | 0.8412 | 0.9517 | 0.7533 |
| fp32 d=64 | fp32 | 0.5092 | 71.5% | 0.6269 | 0.8097 | 0.3213 |

### 128 bytes/vector — 32.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| **remex 1bit d=1024** | remex | **0.7018** | 98.5% | 0.8324 | 0.9550 | 0.7950 |
| remax 1bit uncentered d=1024 | remax-unc | 0.6808 | 95.6% | 0.8080 | 0.9450 | 0.7103 |
| remax 1bit d=1024 | remax | 0.6772 | 95.1% | 0.8136 | 0.9500 | 0.7013 |

### 64 bytes/vector — 64.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| remax 1bit d=512 | remax | 0.6501 | 91.3% | 0.7968 | 0.9417 | 0.5737 |

### 32 bytes/vector — 128.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| remax 1bit d=256 | remax | 0.5722 | 80.3% | 0.7008 | 0.8677 | 0.4077 |

### 16 bytes/vector — 256.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| remax 1bit d=128 | remax | 0.4214 | 59.2% | 0.5363 | 0.7669 | 0.2657 |

### 8 bytes/vector — 512.0x smaller than fp32-full

| codec | family | nDCG@10 | % of fp32 | R@10 | R@100 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| remax 1bit d=64 | remax | 0.2036 | 28.6% | 0.3192 | 0.6200 | 0.1363 |

## What to deploy

Cheapest codec (fewest bytes/vector) still clearing each quality bar:

| bar | codec | bytes/vec | compression | nDCG@10 | % of fp32 | agree@10 |
|---|---|--:|--:|--:|--:|--:|
| >= 99% | remex 3bit d=1024 | 384 | 10.7x | 0.7097 | 99.6% | 0.9360 |
| >= 97% | remex 1bit d=1024 | 128 | 32.0x | 0.7018 | 98.5% | 0.7950 |
| >= 95% | remex 1bit d=1024 | 128 | 32.0x | 0.7018 | 98.5% | 0.7950 |

