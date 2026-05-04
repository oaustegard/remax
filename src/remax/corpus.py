"""remax.corpus — Corpus: paired bit-vector index + SQLite metadata store.

Bridges the gap between raw array indices returned by ``hamming_search``
and the actual documents/records those positions represent.

Schema
------
::

    CREATE TABLE corpus_meta (
      rowid INTEGER PRIMARY KEY,  -- = position in packed array (0-indexed)
      record_id TEXT NOT NULL,    -- external ID (DOI, S2 corpus ID, UUID…)
      meta TEXT                   -- optional JSON payload
    );
    CREATE INDEX idx_record_id ON corpus_meta(record_id);

The SQLite ``rowid`` is the B-tree key, so point lookups by position are
O(log n) at worst and sub-millisecond in practice. The index on
``record_id`` supports reverse lookups ("is this document already indexed?").
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core import SignBitQuantizer

__all__ = ["Corpus", "Result"]

_BIN_NAME = "index.bin"
_DB_NAME = "meta.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_meta (
    rowid    INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL,
    meta      TEXT
);
CREATE INDEX IF NOT EXISTS idx_record_id ON corpus_meta(record_id);
"""


@dataclass(frozen=True)
class Result:
    """A single search result with resolved metadata.

    Attributes
    ----------
    rank : int
        0-based rank in the result list (0 = closest).
    distance : int
        Hamming distance between the query code and this code.
    record_id : str
        External identifier supplied at index build time.
    meta : dict | None
        Optional JSON metadata, or ``None`` if none was stored.
    """

    rank: int
    distance: int
    record_id: str
    meta: Optional[dict]


class Corpus:
    """Paired bit-vector index + SQLite metadata store.

    A ``Corpus`` lives in a directory that contains two files:

    * ``index.bin`` — packed uint8 codes produced by :class:`SignBitQuantizer`.
    * ``meta.db``   — SQLite database mapping row positions to external IDs
      and optional JSON metadata.

    Parameters
    ----------
    path : str | Path
        Directory containing ``index.bin`` and ``meta.db``.

    Examples
    --------
    Build a corpus from scratch::

        import numpy as np
        from remax import Corpus

        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 768))
        ids = [f"doc-{i}" for i in range(1000)]

        c = Corpus.build("my_corpus/", X, ids, d=768, seed=42)

    Search and get back rich results::

        results = c.search(X[0], k=5)
        for r in results:
            print(r.rank, r.distance, r.record_id)

    Reverse lookup::

        pos = c.lookup("doc-42")   # → 42
    """

    def __init__(self, path: str | Path):
        self._dir = Path(path)
        bin_path = self._dir / _BIN_NAME
        db_path = self._dir / _DB_NAME

        if not bin_path.exists():
            raise FileNotFoundError(f"index.bin not found in {self._dir}")
        if not db_path.exists():
            raise FileNotFoundError(f"meta.db not found in {self._dir}")

        raw = np.fromfile(bin_path, dtype=np.uint8)
        # Header: [n (int64), d (int64), seed_present (int8), seed (int64)]
        n = int(raw[0:8].view(np.int64)[0])
        d = int(raw[8:16].view(np.int64)[0])
        seed_present = int(raw[16])
        seed = int(raw[17:25].view(np.int64)[0]) if seed_present else None

        codes_bytes = n * (d // 8)
        codes_flat = raw[25 : 25 + codes_bytes]
        self._codes = codes_flat.reshape(n, d // 8)
        self._quantizer = SignBitQuantizer(d=d, seed=seed)
        self._db_path = str(db_path)

    # ------------------------------------------------------------------ #
    # Class-method constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        path: str | Path,
        vectors: np.ndarray,
        ids: list[str],
        *,
        d: int | None = None,
        seed: int | None = None,
        meta: list[dict] | None = None,
        center: bool = False,
    ) -> "Corpus":
        """Build a corpus from raw vectors and record IDs.

        Parameters
        ----------
        path : str | Path
            Destination directory. Created if it does not exist.
        vectors : np.ndarray, shape (n, d)
            Raw (un-encoded) embeddings. ``d`` must be divisible by 8.
        ids : list[str]
            External record IDs in corpus order. Must have ``len(ids) == n``.
        d : int | None
            Embedding dimension. Inferred from ``vectors.shape[1]`` if omitted.
        seed : int | None
            RNG seed for the Haar rotation.
        meta : list[dict] | None
            Optional per-record JSON-serialisable metadata dicts.
        center : bool, default=False
            Subtract the corpus mean before encoding (mirrors the bench
            harness ``--center`` flag). The mean is *not* stored; callers
            must apply the same centering at query time if they enable this.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        n, inferred_d = vectors.shape
        if d is None:
            d = inferred_d
        elif d != inferred_d:
            raise ValueError(
                f"d={d} does not match vectors.shape[1]={inferred_d}"
            )
        if d % 8 != 0:
            raise ValueError(
                f"d={d} not divisible by 8; remax codes are bit-packed bytewise"
            )
        if len(ids) != n:
            raise ValueError(
                f"len(ids)={len(ids)} must equal number of vectors n={n}"
            )
        if meta is not None and len(meta) != n:
            raise ValueError(
                f"len(meta)={len(meta)} must equal number of vectors n={n}"
            )

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)

        q = SignBitQuantizer(d=d, seed=seed)
        enc_vectors = vectors - vectors.mean(axis=0) if center else vectors
        codes = q.encode(enc_vectors)  # (n, d//8) uint8

        # Write binary index with a small header.
        bin_path = dest / _BIN_NAME
        header = np.zeros(25, dtype=np.uint8)
        header[0:8] = np.array([n], dtype=np.int64).view(np.uint8)
        header[8:16] = np.array([d], dtype=np.int64).view(np.uint8)
        if seed is not None:
            header[16] = 1
            header[17:25] = np.array([seed], dtype=np.int64).view(np.uint8)
        with open(bin_path, "wb") as f:
            f.write(header.tobytes())
            f.write(codes.tobytes())

        # Write SQLite metadata.
        db_path = dest / _DB_NAME
        if db_path.exists():
            db_path.unlink()
        con = sqlite3.connect(str(db_path))
        try:
            con.executescript(_SCHEMA)
            rows = []
            for i in range(n):
                meta_json = json.dumps(meta[i]) if meta is not None else None
                rows.append((i, ids[i], meta_json))
            con.executemany(
                "INSERT INTO corpus_meta (rowid, record_id, meta) VALUES (?, ?, ?)",
                rows,
            )
            con.commit()
        finally:
            con.close()

        return cls(dest)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(self, query: np.ndarray, k: int = 10) -> list[Result]:
        """Return the top-k nearest neighbours with resolved metadata.

        Parameters
        ----------
        query : np.ndarray, shape (d,)
            Raw (un-encoded) query vector.
        k : int
            Number of results to return.

        Returns
        -------
        list[Result]
            Length ``min(k, n)`` results sorted by ascending Hamming distance.
        """
        indices, distances = self._quantizer.search(
            query, self._codes, k=k, return_distances=True
        )
        indices = np.asarray(indices)
        distances = np.asarray(distances)

        positions = indices.tolist()
        placeholders = ",".join("?" * len(positions))
        con = sqlite3.connect(self._db_path)
        try:
            rows = con.execute(
                f"SELECT rowid, record_id, meta FROM corpus_meta "
                f"WHERE rowid IN ({placeholders})",
                positions,
            ).fetchall()
        finally:
            con.close()

        # Build a pos → (record_id, meta) map; the DB may return rows in
        # any order.
        lookup: dict[int, tuple[str, Any]] = {}
        for rowid, record_id, meta_json in rows:
            lookup[rowid] = (record_id, json.loads(meta_json) if meta_json else None)

        results = []
        for rank, (pos, dist) in enumerate(zip(positions, distances.tolist())):
            record_id, meta = lookup.get(pos, (str(pos), None))
            results.append(Result(rank=rank, distance=int(dist), record_id=record_id, meta=meta))
        return results

    def lookup(self, record_id: str) -> int | None:
        """Reverse lookup: external record ID → array position.

        Returns ``None`` if the record_id is not in the corpus.
        """
        con = sqlite3.connect(self._db_path)
        try:
            row = con.execute(
                "SELECT rowid FROM corpus_meta WHERE record_id = ? LIMIT 1",
                (record_id,),
            ).fetchone()
        finally:
            con.close()
        return int(row[0]) if row is not None else None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Number of indexed vectors."""
        return self._codes.shape[0]

    @property
    def d(self) -> int:
        """Embedding dimension."""
        return self._quantizer.d

    @property
    def codes(self) -> np.ndarray:
        """Packed uint8 codes, shape (n, d // 8). Read-only view."""
        return self._codes

    def __repr__(self) -> str:
        return f"Corpus(n={self.n}, d={self.d}, path={str(self._dir)!r})"
