# Postgres metadata recipe

remax is SQLite-native: the `Corpus` class ships a packed `.bin` index and
a `.db` sidecar that live together in a directory.  For teams that already
run Postgres and want to serve metadata from their existing database instead
of shipping SQLite files, this page documents the equivalent schema and query
patterns.

This is a **recipe**, not a runtime dependency.  Nothing in `remax` imports
`psycopg2` or any Postgres driver.

---

## Schema

```sql
CREATE TABLE corpus_meta (
    pos       INTEGER PRIMARY KEY,  -- = position in the packed array (0-indexed)
    record_id TEXT    NOT NULL,
    meta      JSONB                 -- optional metadata; use NULL if not needed
);

CREATE INDEX idx_record_id ON corpus_meta (record_id);
```

`pos` is the same integer that `Corpus.search()` returns as
`Result.rank`'s backing array index.  Using it as the primary key means
point lookups are a single B-tree probe with no join table.

`meta` is `JSONB` rather than `TEXT` so Postgres can index and query fields
inside it with `->` / `@>` operators if needed.

---

## Populating

```python
import psycopg2
import numpy as np
from remax.corpus import Corpus

# Build the remax index as usual.
c = Corpus.build("my_corpus/", vectors, ids, meta=meta_list, seed=42)

# Mirror the metadata into Postgres.
con = psycopg2.connect("dbname=mydb")
cur = con.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS corpus_meta (
        pos INTEGER PRIMARY KEY,
        record_id TEXT NOT NULL,
        meta JSONB
    )
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_record_id ON corpus_meta (record_id)")

import json

rows = [(i, ids[i], json.dumps(meta_list[i]) if meta_list else None)
        for i in range(len(ids))]
cur.executemany(
    "INSERT INTO corpus_meta (pos, record_id, meta) VALUES (%s, %s, %s)",
    rows,
)
con.commit()
cur.close()
con.close()
```

---

## Querying after `hamming_search`

`Corpus.search()` already handles metadata retrieval for the SQLite case.
For Postgres, retrieve raw indices from `SignBitQuantizer.search()` and
then resolve them in a single round-trip using `= ANY`:

```python
import psycopg2
from remax import SignBitQuantizer
import numpy as np

q = SignBitQuantizer(d=768, seed=42)
codes = np.fromfile("my_corpus/index.bin", dtype=np.uint8)[25:].reshape(-1, 96)

# top-K Hamming search — returns raw array positions
indices, distances = q.search(query_vec, codes, k=10, return_distances=True)
top_k = indices.tolist()   # e.g. [4201, 812, 9933, ...]

con = psycopg2.connect("dbname=mydb")
cur = con.cursor()
cur.execute(
    """
    SELECT pos, record_id, meta
    FROM   corpus_meta
    WHERE  pos = ANY(%s)
    ORDER BY array_position(%s, pos)   -- preserve rank order
    """,
    (top_k, top_k),
)
rows = cur.fetchall()
cur.close()
con.close()

for pos, record_id, meta in rows:
    print(pos, record_id, meta)
```

`array_position(%s, pos)` preserves the Hamming-distance rank without a
second sort.  Postgres resolves 500 integer PK lookups in a 100 M-row table
in the microsecond range — the Hamming scan dominates.

---

## Reverse lookup

```sql
-- "Is this document already indexed?"
SELECT pos FROM corpus_meta WHERE record_id = $1 LIMIT 1;
```

The `idx_record_id` index makes this a sub-millisecond seek.

---

## When to choose Postgres over SQLite

| Situation | Recommendation |
|---|---|
| Self-contained file bundle, no external infra | SQLite (`Corpus` built-in) |
| Metadata already lives in Postgres | Postgres recipe (this doc) |
| Metadata needs multi-field JSONB queries | Postgres (`@>`, GIN index) |
| Multiple writers / concurrent index updates | Postgres |
| Embedding count < 50 M, single-process server | Either; SQLite is simpler |

SQLite's write-ahead log (WAL mode) handles concurrent *reads* fine.  Switch
to Postgres when you have concurrent *writes* to metadata or need to JOIN
with other tables in your existing schema.
