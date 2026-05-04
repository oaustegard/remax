## remax stage-2 rerank — sign-bit candidates → cross-encoder vs float32-IP

- **Dataset**: SPECTER2  
- **Corpus / queries**: n_corpus=9900, n_queries=100, d=768
- **Stage 1**: centered 1-bit SimHash, Hamming top-100. Quantizer seed = 42, query split seed = 99.
- **Stage 2a**: float32 inner-product rerank (the baseline — and, under the float32-IP truth metric, the *optimal* reranker over the candidate set).
- **Stage 2b**: cross-encoder rerank — `cross-encoder/ms-marco-MiniLM-L-6-v2` via ONNX Runtime CPU.
- **Metric**: R@10 vs float32 inner-product ground truth on the raw (un-centered) corpus.
- **Latency**: wall-clock per query for stage-2 work only (stage-1 Hamming and one-time cross-encoder model load are excluded).

### Recall and latency

| stage                        | R@10 | latency / query |
|------------------------------|-------|-----------------|
| stage 1 (raw 1-bit Hamming)  | 0.635 | (n/a)           |
| stage 2a (float32-IP rerank) | 0.983 | 0.1 ms          |
| stage 2b (cross-encoder)     | 0.305 | 5065.5 ms       |

**Stage-2 R@10 ceiling (= stage 2a):** 0.983 — the fraction of true top-10 present in the stage-1 candidate set. Float32-IP rerank attains this ceiling exactly (it picks top-10 by descending IP, which is the truth metric). No reranker scoring on a different signal can strictly exceed it.

**Stage-1 R@100 (candidate-set retention):** 0.689 — fraction of the true top-100 that the candidate set covers. Different from the stage-2 R@10 ceiling; reported here for context on stage-1 quality at the wider cut.

### Discussion

**Float32-IP rerank wins decisively.** Stage 1 R@10 of 0.635 climbs to 0.983 after float32-IP rerank of the top-100 candidates — almost all of the recall sign-bit stage 1 left on the table is recoverable, at 0.1 ms per query. The sign-bit + float32-IP-rerank pipeline is essentially a lossless R@10 approximation of full float32 search at this n / top-100.

**Off-the-shelf cross-encoder doesn't.** Stage 2b achieves R@10 = 0.305 at ~5065.5 ms per query — strictly worse than the float32-IP baseline, and 50655× slower. Two confounders worth naming:

1. **Truth-metric mismatch.** R@K is measured against float32 IP ground truth. Float32-IP rerank optimises the truth metric directly; the cross-encoder optimises a *learned* relevance function. To the extent the two disagree, the cross-encoder is wrong by definition under this metric.
2. **Domain mismatch.** `ms-marco-MiniLM-L-6-v2` is trained on short web search query → web document relevance. Both "query" and "document" here are full SPECTER2 paper records (title + abstract, hundreds of tokens). The model is being asked to do paper–paper semantic similarity in a regime it never saw during pretraining.

The result isn't that cross-encoders are bad; it's that the interesting comparison for this corpus needs (a) a domain-matched reranker (a SciBERT-trained CE, or SPECTER2-style bi-encoder distilled into a cross-encoder), or (b) a different truth metric (human relevance judgments rather than float32-IP). Either change would let the cross-encoder express judgments that diverge meaningfully from raw IP.

