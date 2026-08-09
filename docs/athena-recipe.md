# Athena + Parquet recipe

remax's scan is a local one: `Corpus` loads or mmaps `index.bin` and XORs it
against a query code. This page documents the same scan run **as a SQL query over
Parquet on S3**, so a corpus that already lives in S3 can be searched without
standing up a server to hold it.

This is a **recipe**, not a runtime dependency. Nothing in `remax` imports
`pyarrow`, `boto3`, or anything AWS; every snippet below runs in your code,
not the library's. Its companion is
[`postgres-recipe.md`](postgres-recipe.md), which does the same for metadata.

What you are buying, stated plainly up front: Athena reads the entire index
on every query, because an exhaustive Hamming scan has no predicate to prune
on. At 100 M vectors × 256 dims that is **~3.7 GB scanned, about $0.018 per
query** at the $5/TB list price. The same 3.2 GB mmapped locally is free and
sub-second. Athena wins when the corpus already lives in S3 and you would
rather not run the process that holds it — not on cost per query, and not on
latency. The [When to choose Athena](#when-to-choose-athena) table at the
bottom is the short version.

---

## Schema

```sql
CREATE EXTERNAL TABLE specter2_codes (
    pos       BIGINT,   -- array position; the same integer as Corpus's rowid
    record_id STRING,
    c0        BIGINT,   -- big-endian int64 limbs of the packed sign-bit code
    c1        BIGINT,
    c2        BIGINT,
    c3        BIGINT
)
STORED AS PARQUET
LOCATION 's3://my-bucket/specter2/codes/';
```

One row per vector, `ceil(d / 64)` limb columns — four for `d = 256`, twelve
for `d = 768`, and `k` times that for a `StackedSignBitQuantizer` code.

### Why `BIGINT` limbs and not `BINARY(32)`

A single `code BINARY(32)` column is the obvious schema and it does not work.
Trino's bitwise functions — `bit_count(x, bits)`, `bitwise_xor(x, y)` — are
[documented](https://trino.io/docs/current/functions/bitwise.html) as taking
`bigint` and only `bigint`; none of them accept `varbinary`, and Athena
engine v3 is Trino-based, so neither does Athena. `hamming_distance()` exists
but counts differing *characters* in a string, not differing bits. There is
no varbinary path to a popcount.

Splitting the code into 64-bit limbs and summing per-limb popcounts uses only
functions that have been in Presto and Trino for years:

```
bit_count(bitwise_xor(c0, q0), 64) + bit_count(bitwise_xor(c1, q1), 64) + …
```

`bit_count(x, 64)` counts set bits in the two's-complement 64-bit
representation, so negative limbs are counted correctly and no unsigned type
is needed. The limbs are read big-endian purely as a convention: Hamming
distance does not care what order the bits arrive in, only that the corpus
side and the query side agree, so the one rule is that both go through the
same `to_limbs` below.

Keep `pos` even though `record_id` is the interesting column. It is the array
position, identical to `Corpus`'s SQLite `rowid` and to the `pos` in the
Postgres recipe, which makes stage-2 lookups against a fixed-stride float
blob a matter of arithmetic rather than another index.

---

## The shared transform

Both sides — the Parquet you write and the constants you interpolate into the
query — must map bytes to limbs identically. Get this wrong in one place and
nothing errors: the query returns 100 rows, ranked by a distance to a
different vector than the one you asked about. One function, used twice:

```python
# remax_athena.py
import numpy as np


def to_limbs(codes: np.ndarray) -> np.ndarray:
    """Packed remax codes -> big-endian int64 limbs.

    ``codes`` is ``(n, B)`` uint8 as produced by ``SignBitQuantizer.encode``
    or read from ``Corpus.codes``; a 1-D ``(B,)`` query code is accepted as a
    single row. Returns ``(n, ceil(B / 8))`` int64.

    Codes whose byte length is not a multiple of 8 are zero-padded on the
    right. The padding is identical on both sides and 0 XOR 0 contributes no
    set bits, so distances are unaffected.
    """
    codes = np.ascontiguousarray(np.atleast_2d(codes), dtype=np.uint8)
    n, nbytes = codes.shape
    pad = -nbytes % 8
    if pad:
        codes = np.hstack([codes, np.zeros((n, pad), np.uint8)])
    return codes.reshape(n, -1, 8).view(">i8").reshape(n, -1).astype(np.int64)
```

---

## Encode side: dump a corpus to Parquet

`Corpus` already holds the codes; the record IDs come from its SQLite sidecar,
whose schema is documented in `src/remax/corpus.py`. Reading them in `rowid`
order is what aligns them with the code rows.

<!-- athena-exec: skip -->
```python
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from remax import Corpus
from remax_athena import to_limbs


def record_ids(corpus_dir: str) -> list[str]:
    """Record IDs in array-position order."""
    con = sqlite3.connect(f"file:{corpus_dir}/meta.db?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT record_id FROM corpus_meta ORDER BY rowid")]
    finally:
        con.close()


def export_parquet(dest: str, codes: np.ndarray, ids: list[str],
                   *, row_group_size: int = 1_000_000) -> int:
    limbs = to_limbs(codes)
    n, n_limbs = limbs.shape
    if len(ids) != n:
        raise ValueError(f"{len(ids)} record IDs for {n} codes")

    limb_cols = [f"c{j}" for j in range(n_limbs)]
    table = pa.table({
        "pos": pa.array(np.arange(n), pa.int64()),
        "record_id": pa.array(ids, pa.string()),
        **{c: pa.array(limbs[:, j], pa.int64())
           for j, c in enumerate(limb_cols)},
    })
    pq.write_table(
        table, dest,
        row_group_size=row_group_size,
        use_dictionary=False,
        # Sign bits are incompressible by construction — see "File layout".
        compression={"pos": "zstd", "record_id": "zstd",
                     **{c: "none" for c in limb_cols}},
    )
    return n_limbs


corpus = Corpus("my_index")
export_parquet("codes-0000.parquet", corpus.codes, record_ids("my_index"))
```

Upload the resulting files under the table's `LOCATION` prefix.

Stacked codes need no special handling: `StackedSignBitQuantizer(d=128, k=3)`
emits 48-byte codes, `to_limbs` turns those into six columns, and the query
grows to six terms. `Corpus` itself is 1-bit only, so for a stacked
index you supply `codes` and `ids` yourself rather than reading them from a
corpus directory.

---

## Query template

Encode the query locally, using the same `(d, seed, rotation)` and corpus mean
the index was built with:

<!-- athena-exec: skip -->
```python
import numpy as np

from remax import Corpus, SignBitQuantizer

SEED = 42          # the seed you passed to Corpus.build

corpus = Corpus("my_index")
q = SignBitQuantizer(d=corpus.d, seed=SEED, rotation=corpus.rotation)


def encode_query(vec: np.ndarray) -> np.ndarray:
    if corpus.centered:
        vec = vec - corpus.mean
    return q.encode(vec)
```

Then interpolate the query's limbs into the SQL as literals:

```python
import numpy as np

from remax_athena import to_limbs


def hamming_sql(query_code: np.ndarray, table: str, k: int = 100) -> str:
    limbs = to_limbs(query_code)[0]
    expr = "\n           + ".join(
        f"bit_count(bitwise_xor(c{j}, {int(v)}), 64)"
        for j, v in enumerate(limbs)
    )
    return (f"SELECT pos, record_id,\n"
            f"       {expr} AS hamming_dist\n"
            f"FROM   {table}\n"
            f"ORDER BY hamming_dist\n"
            f"LIMIT  {k}")
```

For `d = 256` that produces four terms:

```sql
SELECT pos, record_id,
       bit_count(bitwise_xor(c0, 1070409410959393341), 64)
           + bit_count(bitwise_xor(c1, -3748676030185764978), 64)
           + bit_count(bitwise_xor(c2, 1482833071192632394), 64)
           + bit_count(bitwise_xor(c3, 7714125034190845028), 64) AS hamming_dist
FROM   specter2_codes
ORDER BY hamming_dist
LIMIT  100
```

`ORDER BY … LIMIT` is a distributed top-N in Trino, not a sort of the whole
table: each worker keeps its own 100-element heap and only those are merged.
The full scan is the cost; the ranking is not.

Ties are broken arbitrarily, whereas `Corpus.search` breaks them by ascending
position (`packing.stable_top_k`). Add `pos` as a second sort key if you need
the two to agree row-for-row at a tie boundary:

```sql
ORDER BY hamming_dist, pos
```

### Getting the query encoder right

The three things that must match the build are the ones a wrong answer will
not tell you about, because a mis-encoded query produces a perfectly
well-formed ranking of the wrong neighbours:

| | Where it lives | If you get it wrong |
|---|---|---|
| `seed` | the `index.bin` header, or whatever you passed to `Corpus.build` | a different rotation; distances are noise |
| `rotation` | `rotation.json` beside the index — `corpus.rotation` reads it | haar and rht give different codes from the same `(d, seed)` |
| corpus mean | `mean.npy` beside the index — `corpus.mean` reads it | large recall loss (−0.324 R@100 at k=64 on SPECTER2) |

`Corpus` resolves the last two for you, which is why the snippet above opens
the corpus directory even though the scan happens in Athena. Only the seed has
no public accessor. If you no longer have it, it is in the header:

<!-- athena-exec: skip -->
```python
import numpy as np

header = np.fromfile("my_index/index.bin", dtype=np.uint8, count=32)
seed = int(header[24:32].view("<i8")[0]) if header[5] else None
```

Ship `mean.npy` and `rotation.json` wherever the query encoder runs. The
Parquet holds the codes; those two files are what makes a query comparable to
them.

---

## File layout and partitioning

**Do not partition.** Partitioning earns its keep by letting a predicate skip
files, and this query has no predicate — every row is scored, by design. A
hash partition would add planning overhead and prune nothing.

What does matter:

**Column projection.** The stage-1 query touches `record_id` and the limb
columns, and Parquet stores each column as its own byte range, so nothing
else is read. That means a float32 payload column can sit in the same table
without stage 1 paying a byte for it. Dropping `record_id` from the projection
and resolving IDs locally from `pos` saves another ~4 B/row.

**Compression, but only where it does anything.** Measured over 200 000 rows
at `d = 256` with scattered 9-digit record IDs:

| column | plain | zstd |
|---|---|---|
| `pos` | 8.00 B/row | 1.03 B/row |
| `record_id` | 12.67 B/row | 4.80 B/row |
| `c0`–`c3` (the code) | 32.02 B/row | 32.02 B/row |
| **stage-1 projection** | **44.68 B/row** | **36.82 B/row** |

The code columns are sign bits of rotated embeddings — as close to
incompressible as data gets. Over that 200 000-row group they are 6,403,080
bytes plain and 6,403,680 zstd: 600 bytes *larger* compressed, which is the
frame overhead. Compressing them spends CPU on both write and read to save
nothing, so the snippet above sets `"none"` for the limbs and `zstd` for the
two columns that do compress. Athena bills the compressed bytes, so this is
directly ~18% off every query.

Dictionary encoding is off for the same reason in reverse: with 100 M distinct
record IDs, no value repeats, and the dictionary is pure overhead (it measured
*worse* than plain).

**Row groups and file sizes.** Athena splits Parquet at row-group boundaries,
so a single file with many row groups still parallelises — but many small
files do not, because each one carries its own footer read and planning cost.
Target row groups of 1–4 M rows (~32–128 MB of codes at `d = 256`) and files
in the 512 MB–1 GB range. At 100 M vectors that is roughly 6–12 files.

---

## Stage 2: rerank the candidates

Athena gives you 100 `(pos, record_id)` pairs. The rerank against full-precision
vectors should happen somewhere else, and the reason is arithmetic: a `WHERE
record_id IN (…)` against a float32 column in the same table scans that column
in full. At 100 M × 768 × 4 bytes that is 307 GB — about $1.50 per rerank,
some eighty times the stage-1 query it is refining. Row-group statistics
do not rescue it, because 100 candidates scattered across the corpus touch
essentially every row group.

Three places to put it instead, in rough order of how little infrastructure
they need:

**A flat float32 blob on S3, addressed by `pos`.** Write the vectors as a
single fixed-stride file and fetch each candidate with a ranged `GET`. This is
why `pos` is in the schema.

<!-- athena-exec: skip -->
```python
import numpy as np

D, ITEMSIZE = 768, 4

def fetch_vector(s3, bucket: str, key: str, pos: int) -> np.ndarray:
    start = pos * D * ITEMSIZE
    body = s3.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes={start}-{start + D * ITEMSIZE - 1}",
    )["Body"].read()
    return np.frombuffer(body, dtype="<f4")

candidates = [...]                        # pos values from the Athena result
mat = np.stack([fetch_vector(s3, "my-bucket", "specter2/vectors.f32", p)
                for p in candidates])
scores = mat @ query_full                 # query at full d, no truncation
top10 = [candidates[i] for i in np.argsort(-scores)[:10]]
```

100 ranged GETs, issued concurrently, cost fractions of a cent and no index at
all. Point lookups are the whole workload here, which is exactly what object
storage is good at and what a scan engine is not.

**[S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)**
if you want the vectors queryable in their own right rather than only
addressable by position.

**Postgres**, if metadata already lives there — see
[`postgres-recipe.md`](postgres-recipe.md), whose `pos` column is the same
integer as this table's.

Whichever you pick, the rerank itself is a dot product;
[`specter2-search-pipeline.md`](specter2-search-pipeline.md) §4 has the
measured recovery (R@10 = 0.983 from a 100-candidate stage 1 on the v0.1.0
bench).

---

## When to choose Athena

| Situation | Recommendation |
|---|---|
| Corpus fits in RAM on a box you already run | `Corpus` with `residency="mmap"` — free, sub-second |
| Corpus already in S3, no server to hold it, low query volume | Athena (this doc) |
| High query volume against a fixed corpus | A held index; Athena's per-query scan cost does not amortise |
| Need the codes joinable against other tables in Glue | Athena |
| Corpus too large for one box, and a server is acceptable | Shard `Corpus` across processes before reaching for a scan engine |

The dividing line is query volume, not corpus size. Athena's cost is per
query and never goes down; a resident index costs memory once and then serves
queries for nothing. At a handful of queries a day the arithmetic is
overwhelmingly on Athena's side, and it inverts long before a busy service.

---

## What is and is not verified here

The limb arithmetic is verified: `tests/test_athena_recipe.py` extracts
`to_limbs` and `hamming_sql` from this document, evaluates the generated
expression against a real corpus under Trino's `bit_count`/`bitwise_xor`
semantics, and asserts the ranking is identical to `Corpus.search`. If the
transform in this file drifts, that test fails.

Not verified: nothing here has been run against a live Athena endpoint. The
SQL is grounded in the Trino function reference rather than in an executed
query, the byte-per-row figures are measured from pyarrow rather than from an
Athena bill, and the 100 M-scale claims are extrapolated from the same
9 900-vector bench everything else in this repository is measured on
(`bench/results/BASELINE.md`). Real-scale validation is tracked separately in
[#12](https://github.com/oaustegard/remax/issues/12).
