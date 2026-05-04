# Text-to-Candidates Search Pipeline (SPECTER2 + remax)

End-to-end example: take a natural-language query, retrieve 100 candidate
papers from a SPECTER2-embedded corpus using remax as the Stage 1 binary
filter.

## Prerequisites

```bash
pip install remax transformers torch numpy
```

## 1. Encode Your Corpus (offline, once)

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# --- Load SPECTER2 ---
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", set_active=True)
model.eval()


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Mean-pool SPECTER2 embeddings for a list of texts."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


# SPECTER2 expects "title [SEP] abstract" for papers.
# Plain queries work too — the encoder handles both.
corpus_texts = [
    "Attention Is All You Need [SEP] We propose a new simple network architecture...",
    "BERT: Pre-training of Deep Bidirectional Transformers [SEP] We introduce...",
    # ... your full corpus
]
corpus_ids = ["arxiv:1706.03762", "arxiv:1810.04805", ...]  # external IDs

corpus_vectors = encode_texts(corpus_texts)  # (n, 768) float32
```

## 2. Build the Corpus (offline, once)

```python
from remax import Corpus

# Build the index. center=True subtracts the corpus mean before
# sign-packing AND persists the mean as mean.npy so that search()
# can auto-center queries. This is load-bearing for retrieval
# quality. (Raw SPECTER2: R@10 ≈ 0.47; centered: R@10 ≈ 0.64
# on the v0.1.0 bench.)
corpus = Corpus.build(
    "my_index",
    vectors=corpus_vectors,
    ids=corpus_ids,
    seed=42,
    center=True,
    meta=[{"title": t.split(" [SEP] ")[0]} for t in corpus_texts],
)
# Writes:
#   my_index/index.bin  — 32-byte header + packed sign-bit codes
#   my_index/meta.db    — SQLite: rowid → record_id + JSON meta
#   my_index/mean.npy   — corpus mean vector (only when center=True)
```

On-disk footprint: 768 / 8 = **96 bytes per vector** plus the
32-byte header. A 1M-paper corpus is ~92 MB of codes.

## 3. Search at Query Time

```python
from remax import Corpus

# Load (once per process)
corpus = Corpus("my_index")

# Encode the query with the same encoder
query_text = "efficient methods for training large language models"
query_vec = encode_texts([query_text])[0]  # (768,) float32

# Search — Corpus auto-centers the query using the stored mean.
# No manual centering needed.
results = corpus.search(query_vec, k=100)

for r in results[:5]:
    print(f"  #{r.rank}  d={r.distance}  {r.record_id}  {r.meta}")
```

Output (illustrative):
```
  #0  d=287  arxiv:2005.14165  {'title': 'Language Models are Few-Shot Learners'}
  #1  d=291  arxiv:2302.13971  {'title': 'LLaMA: Open and Efficient Foundation...'}
  #2  d=294  arxiv:2203.15556  {'title': 'Training Compute-Optimal Large Language...'}
  #3  d=296  arxiv:1706.03762  {'title': 'Attention Is All You Need'}
  #4  d=298  arxiv:2307.09288  {'title': 'Llama 2: Open Foundation and Fine-Tuned...'}
```

Distances are Hamming (number of bit disagreements out of 768).
Lower = more similar. Theoretical range: 0 to 768.

## 4. Stage 2 Rerank (optional)

The 100 candidates from remax are coarse. For a production pipeline,
rerank them with full-precision scores — either via remex or a
cross-encoder:

```python
# Option A: remex asymmetric inner-product rerank
from remex import Quantizer

q = Quantizer(d=768, bits=4, seed=42)
compressed = q.encode(corpus_vectors[candidate_indices])
final_idx, scores = q.search(compressed, query_vec, k=10)

# Option B: cross-encoder rerank (highest quality, slowest)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query_text, corpus_texts[i]) for i in candidate_indices]
scores = reranker.predict(pairs)
top10 = np.argsort(-scores)[:10]
```

## Notes

**Centering matters.** SPECTER2 embeddings have non-zero per-dimension
means (one dimension has mean ≈ 15.5). Charikar SimHash assumes
mean-zero inputs. Centering closes the gap between naive sign-bit
encoding and Lloyd-Max 1-bit quantization, which adaptively places
its boundary at the per-dimension mean.

**Centering is automatic.** `Corpus.build(center=True)` persists the
corpus mean as `mean.npy`. On load, `Corpus` detects the mean file
and `search()` subtracts it from queries before encoding. Callers
pass raw encoder output — no manual centering step. The mean is
accessible via `corpus.mean` if needed externally (e.g., for
interop with the low-level `SignBitQuantizer` API).

**Query and document share one encoder.** SPECTER2 is a symmetric
(bi-)encoder — queries and documents are embedded with the same model
and adapter. This makes the remax pipeline straightforward: encode
text, pass to `search()`, done.

**Scaling.** At 100M vectors (e.g., the Semantic Scholar corpus), the
full index is ~9.2 GB of packed codes. A brute-force Hamming scan is
feasible on a single machine with NumPy. For sub-linear access, pair
with remex's `IVFCoarseIndex` for cell routing (not yet integrated;
tracked in the remax roadmap).
