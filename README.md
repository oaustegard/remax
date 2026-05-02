# remax

**Rank-correct cosine LSH with a stacked-precision ladder.**

A focused library for one job: 1-bit cosine quantization that preserves *rank*, with a precision ladder that scales linearly in storage and monotonically in recall. Built as a deliberate counterpoint to MSE-optimal scalar quantization, motivated by an empirical inversion observed on real SPECTER2 embeddings.

The name is a pun. Lloyd-Max minimizes reconstruction error. remax targets rank.

## Why this exists

In [*One Bit Beats Two*](https://muninn.austegard.com/blog/one-bit-beats-two.html), 1-bit Matryoshka extraction from a Lloyd-Max code beat both 2-bit and 3-bit on R@10 — by 13 and 4 percentage points respectively — on real SPECTER2 embeddings. The 1-bit code is bit-for-bit identical to Charikar's 2002 SimHash. Its 2- and 3-bit cousins inherit Lloyd-Max's MSE-optimal interior boundaries, which are wrong for ranking: an interior bin flip changes the dot-product sign without proportional MSE penalty.

The fix Lloyd-Max can't deliver: don't refine each coordinate, *stack independent SimHashes*. k stacked sign-bit signatures give k bits per dimension with rank-correct semantics at every step (variance ∝ 1/k), and no broken middle.

remax is the library that does that, exclusively.

## Status

v0.0.0 — repo scaffolded, work tracked in [issues](https://github.com/oaustegard/remax/issues). See the [epic](https://github.com/oaustegard/remax/issues/1) for the v0.1.0 plan.

## Relationship to remex

[remex](https://github.com/oaustegard/remex) is the multi-precision Lloyd-Max + Matryoshka library it shares lineage with. remex is a Swiss Army knife optimized for storage MSE, with rank-correct 1-bit *as a free MSB extraction*. remax is a chisel optimized for rank, exclusively.

The two coexist:
- Use **remex** when you need a single 8-bit storage tier with cheap Matryoshka extraction down to 1 bit.
- Use **remax** when you need a pure-rank in-memory tier with a precision ladder that doesn't break in the middle.
- Use **both** if your two-stage retrieval architecture wants a remax-ladder Stage 1 and remex Stage 2.

## License

MIT, © 2026 Oskar Austegard.
