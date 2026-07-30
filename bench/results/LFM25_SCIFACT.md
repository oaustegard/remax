# LFM2.5 codec bake-off — BEIR SciFact

Corpus 5183 docs / 300 queries / 339 judgments. Seed 42.

## Geometry — LFM2.5-Embedding-350M

| metric | value | isotropic reference |
|---|--:|--:|
| n x d | 5183 x 1024 | — |
| norm mean | 1.0000 | — |
| norm CV (std/mean) | 0.0000 | high = varied lengths |
| off-center ratio | 0.3446 | 0 = centered at origin |
| top-1 eigenvalue frac | 0.0522 | 0.0010 |
| top-10 eigenvalue frac | 0.2068 | 0.0098 |
| participation ratio | 118.3 / 1024 | 1024 |
| post-rotation sigma | 0.02928 | 0.03125 |
| **post-rotation sigma ratio** | **0.9370** | **1.0000** |
| post-rotation kurtosis | -0.010 | 0.000 |
| KS reject frac | 0.00 | 0.00 |

**Regime: isotropic** (sigma ratio 0.937)

Prediction: remex dominates at every matched byte budget; 4-bit near-lossless

Centering expected to matter: True

## Results (sorted by bytes/vector)

| codec | B/vec | ratio | nDCG@10 | dnDCG | R@10 | R@100 | agree@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| remax 1bit d=64 | 8 | 512.0x | 0.2036 | -0.5087 | 0.3192 | 0.6200 | 0.1363 |
| remax 1bit d=128 | 16 | 256.0x | 0.4214 | -0.2908 | 0.5363 | 0.7669 | 0.2657 |
| remax 1bit d=256 | 32 | 128.0x | 0.5722 | -0.1400 | 0.7008 | 0.8677 | 0.4077 |
| remax 1bit d=512 | 64 | 64.0x | 0.6501 | -0.0622 | 0.7968 | 0.9417 | 0.5737 |
| remex 1bit d=1024 | 128 | 32.0x | 0.7018 | -0.0104 | 0.8324 | 0.9550 | 0.7950 |
| remax 1bit uncentered d=1024 | 128 | 32.0x | 0.6808 | -0.0315 | 0.8080 | 0.9450 | 0.7103 |
| remax 1bit d=1024 | 128 | 32.0x | 0.6772 | -0.0350 | 0.8136 | 0.9500 | 0.7013 |
| remex 2bit d=1024 | 256 | 16.0x | 0.7042 | -0.0081 | 0.8324 | 0.9550 | 0.8830 |
| remax k=2 d=1024 | 256 | 16.0x | 0.6989 | -0.0133 | 0.8412 | 0.9517 | 0.7533 |
| fp32 d=64 | 256 | 16.0x | 0.5092 | -0.2031 | 0.6269 | 0.8097 | 0.3213 |
| remex 3bit d=1024 | 384 | 10.7x | 0.7097 | -0.0025 | 0.8358 | 0.9617 | 0.9360 |
| remex 4bit d=1024 | 512 | 8.0x | 0.7110 | -0.0012 | 0.8336 | 0.9583 | 0.9617 |
| remax k=4 d=1024 | 512 | 8.0x | 0.7078 | -0.0044 | 0.8391 | 0.9633 | 0.7880 |
| fp32 d=128 | 512 | 8.0x | 0.6085 | -0.1037 | 0.7353 | 0.8993 | 0.4797 |
| remex 8bit d=1024 | 1024 | 4.0x | 0.7122 | +0.0000 | 0.8336 | 0.9617 | 0.9960 |
| int8 d=1024 | 1024 | 4.0x | 0.7122 | +0.0000 | 0.8336 | 0.9617 | 0.9960 |
| remax k=8 d=1024 | 1024 | 4.0x | 0.7087 | -0.0035 | 0.8369 | 0.9567 | 0.8190 |
| fp32 d=256 | 1024 | 4.0x | 0.6572 | -0.0551 | 0.7861 | 0.9450 | 0.6380 |
| fp32 d=512 | 2048 | 2.0x | 0.7042 | -0.0080 | 0.8386 | 0.9600 | 0.7880 |
| fp32 d=1024 | 4096 | 1.0x | 0.7122 | +0.0000 | 0.8336 | 0.9617 | 1.0000 |
