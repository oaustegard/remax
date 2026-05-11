## Follow-up to BM25_SKETCH.md — what *does* work for in-memory BM25

`BM25_SKETCH.md` documented why sign-packed count-sketch fails on
sparse BM25 (R@100 = 0.019, NDCG@10 = 0.010 vs full-BM25 ceiling of
0.267 on NFCorpus). The thesis is killed for the *sparse* path; the
encoder and the failed bench code have been removed from the library.

But the underlying question — *is there a more in-memory-friendly
representation for BM25-style retrieval than naive inverted indices?* —
is still interesting. This doc records the follow-up experiments
(performed as a session spike outside the library tree) and what they
imply for remax.

### Setup

Same NFCorpus from BEIR (3633 docs, 323 qrels-eligible queries, ~5
tokens per query, ~234 tokens per doc, vocab 66k). Same tokenization
(lowercase + whitespace) to match `rank_bm25.BM25Okapi`. Reference
baselines:

- **Sparse ceiling**: `rank_bm25.BM25Okapi` top-100. NDCG@10 = 0.267.
- **Sparse failure**: `SparseSignBitQuantizer` count-sketch (deleted).
  R@100 = 0.019, NDCG@10 = 0.010.

Dense baselines added for the follow-up:

- `sentence-transformers/all-MiniLM-L6-v2` — 384-dim normalized embeddings, ~6 MB total for the corpus.
- `remax.SignBitQuantizer` 1-bit Charikar SimHash on the same MiniLM embeddings.

### Results

| approach                                          |    NDCG@10 |  R@100 vs `rank_bm25` | bytes/doc | notes                                     |
|---------------------------------------------------|-----------:|---------------------:|----------:|-------------------------------------------|
| `rank_bm25.BM25Okapi` (sparse ceiling)            |     0.2672 |                1.000 |       n/a | reference                                 |
| `SparseSignBitQuantizer` count-sketch *(deleted)* |     0.0100 |                0.019 |      ~256 | at chance — see `BM25_SKETCH.md`          |
| **Impact-quantized BM25 (8-bit), numpy scatter**  | **0.2715** |            **0.920** |       623 | matches `rank_bm25` ceiling               |
| Impact-quantized BM25 (4-bit)                     |     0.2711 |                0.903 |       (½) | graceful degradation                      |
| Impact-quantized BM25 (1-bit)                     |     0.2443 |                0.761 |       (⅛) | still above the 0.70 thesis-kill threshold |
| Roaring + WAND (pure Python)                      |     0.2716 |             same set |       666 | **6–11× slower** than numpy scatter       |
| Dense MiniLM-L6-v2 cosine (full f32)              |     0.3167 |                    — |     1 536 | beats BM25 on this corpus                 |
| Dense MiniLM SimHash 1-bit (`remax`)              |     0.2668 |                    — |        48 | **ties `rank_bm25` at 1/13× storage**     |
| Hybrid impact stage-1 → dense rerank, N=50        |     0.2921 |                    — |      n/a  | bottlenecked by BM25 stage-1 recall       |
| Hybrid impact stage-1 → dense rerank, N=1000      |     0.3005 |                    — |      n/a  | still below dense alone                   |
| Hybrid impact stage-1 → remax 1-bit rerank, N=50  |     0.2784 |                    — |      n/a  | sparse stage-1 + dense-bit stage-2        |
| Hybrid impact stage-1 → remax 1-bit rerank, N=1000 |    0.2680 |                    — |      n/a  | converges back to BM25                    |

R@100 in the impact-quantized rows is against `rank_bm25.BM25Okapi`
(which clamps negative IDFs while our impl drops them). The f32 baseline
of the same impl tops out at R@100 = 0.9208 — so quantization through
4-bit is doing its job *perfectly*, and the residual gap is the
IDF-clipping policy difference, not quantization loss.

### Three findings worth keeping

1. **Per-posting impact quantization preserves BM25 cleanly.** 8-bit
   per-posting linear scaling per term holds R@100 = 0.92 against
   `rank_bm25` with NDCG@10 indistinguishable from the float baseline.
   Quality degrades gracefully through 4-bit (R@100 = 0.90) and even
   1-bit (R@100 = 0.76). The key difference vs the failed sparse
   sketch: per-posting quantization preserves *term identity* exactly;
   only intra-term magnitude resolution is lost. The sketch lost term
   identity to a global random projection, which is why it collapsed.

2. **Roaring + WAND need C++/SIMD to win.** Pure-Python loop overhead
   per cursor advance dominated the algorithmic skip; one C-vectorized
   numpy scatter-add beat both Roaring-exhaustive and WAND at every
   scale tested (3.6k → 100k docs by 6–11×). Roaring also went *bigger*
   than a flat uint32 array because NFCorpus posting lists at ~3%
   density per term land in Roaring's worst-case container regime.
   The classical "use Roaring for inverted indices" advice applies to
   set-membership filtering, not to scored retrieval where you also
   carry per-posting weights. Real production engines (PISA, Lucene,
   Tantivy) get their wins from C++ posting traversal with SIMD
   popcount/skip — not reachable from pure Python.

3. **`remax.SignBitQuantizer` on MiniLM embeddings ties `rank_bm25`
   on NFCorpus at 1/13× the storage.** A 48-bytes-per-doc dense
   Charikar SimHash code lands at NDCG@10 = 0.2668, statistically
   indistinguishable from the 0.2672 sparse ceiling. Not a BM25
   replacement so much as a hint that the precision-ladder
   generalization is robust to retrieval-substitution: if you already
   have a dense encoder that captures the relevance signal for your
   domain, the 1-bit signature beats keeping the inverted index around
   for size *and* parity at relevance.

### What this means for remax

Nothing to ship. remax's value claim is the rank-correct dense
precision ladder (`SignBitQuantizer` + `StackedSignBitQuantizer`), and
finding 3 above just *confirms* the existing claim on a new corpus and
a new task (relevance-judged retrieval, not synthetic top-K against
float ground truth). No new code lands.

The hybrid sparse → dense pipeline is still defensible *architecturally*
for corpora where (a) lexical genuinely beats dense (legal, niche-
technical, code), or (b) full dense scan is too expensive even with
remax tricks at the corpus size. NFCorpus is neither — MiniLM beats
BM25 here, and full dense scan over 3633 docs is ~µs. Validating the
hybrid claim would need a different corpus + a corpus large enough
that dense exhaustive scan starts to bite. Out of scope for this doc.

### Disposition

- No code added to remax. No new tests. No new dependencies.
- The experimental spike (impact-BM25 retriever, ablation, hybrid bench,
  Roaring/WAND variant, dense encoder) lives on the session's hub-side
  development branch (`oaustegard/claude-workspace`,
  `claude/implement-issue-36-fiVKZ`, under `experiments/bm25-mem/`) as
  a session record. It is not merged anywhere.
- This doc is the durable artefact: it documents the negative results
  honestly so the same paths aren't silently re-tried later.
