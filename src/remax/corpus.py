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

Binary file format
------------------
``index.bin`` starts with a 32-byte header (all fields little-endian)::

    bytes  0..3   magic         b'RMAX'
    byte   4      version       0x01
    byte   5      seed_present  0x00 / 0x01
    bytes  6..7   reserved      \x00\x00
    bytes  8..15  n             int64 (number of rows)
    bytes 16..23  d             int64 (dimension)
    bytes 24..31  seed          int64 (zero when seed_present == 0)

Codes follow as ``n * (d // 8)`` bytes of packed signs.

A v0 reader (no magic, 25-byte header) is supported in :meth:`Corpus.__init__`
so older indexes still load with a deprecation warning.
"""

from __future__ import annotations

import json
import os
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from .core import SignBitQuantizer

__all__ = ["Corpus", "Result"]

_BIN_NAME = "index.bin"
_DB_NAME = "meta.db"
_MEAN_NAME = "mean.npy"

# Format constants
_MAGIC = b"RMAX"
_VERSION = 1
_HEADER_LEN = 32          # v1 header
_LEGACY_HEADER_LEN = 25   # v0 (no magic)

# SQLite IN(...) parameter limit. Modern sqlite (3.32+) raises this to 32766,
# but older builds cap at 999. We chunk to stay below the conservative limit.
_SQLITE_IN_CHUNK = 900

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
    """A single search result with resolved metadata."""

    rank: int
    distance: int
    record_id: str
    meta: Optional[dict]


def _validate_dims(n: int, d: int) -> None:
    """Sanity-check header dims so a corrupt/hostile file fails fast."""
    if n < 0:
        raise ValueError(f"corrupt index: n={n} (must be >= 0)")
    if d <= 0:
        raise ValueError(f"corrupt index: d={d} (must be > 0)")
    if d % 8 != 0:
        raise ValueError(
            f"corrupt index: d={d} not divisible by 8 (codes are bit-packed)"
        )
    if n > (1 << 48):
        raise ValueError(f"corrupt index: n={n} implausibly large")


def _read_header(raw: np.ndarray) -> tuple[int, int, int | None, int]:
    """Parse the bin header. Returns (n, d, seed, payload_offset)."""
    if raw.size >= _HEADER_LEN and bytes(raw[:4]) == _MAGIC:
        version = int(raw[4])
        if version != _VERSION:
            raise ValueError(
                f"unsupported index format version {version} "
                f"(this build understands v{_VERSION})"
            )
        seed_present = int(raw[5])
        n = int(raw[8:16].view("<i8")[0])
        d = int(raw[16:24].view("<i8")[0])
        seed = int(raw[24:32].view("<i8")[0]) if seed_present else None
        _validate_dims(n, d)
        return n, d, seed, _HEADER_LEN

    # Legacy v0: 25-byte header, no magic.
    if raw.size < _LEGACY_HEADER_LEN:
        raise ValueError(
            f"index.bin too short ({raw.size} bytes); not a remax index"
        )
    warnings.warn(
        "Loading legacy v0 index.bin (no magic bytes). Re-run "
        "Corpus.build() to upgrade to v1; v0 support will be removed.",
        DeprecationWarning,
        stacklevel=3,
    )
    n = int(raw[0:8].view("<i8")[0])
    d = int(raw[8:16].view("<i8")[0])
    seed_present = int(raw[16])
    seed = int(raw[17:25].view("<i8")[0]) if seed_present else None
    _validate_dims(n, d)
    return n, d, seed, _LEGACY_HEADER_LEN


def _write_header(n: int, d: int, seed: int | None) -> bytes:
    """Build the v1 binary header. 32 bytes, little-endian."""
    _validate_dims(n, d)
    buf = bytearray(_HEADER_LEN)
    buf[0:4] = _MAGIC
    buf[4] = _VERSION
    buf[5] = 0 if seed is None else 1
    # bytes 6..7 reserved (zero)
    buf[8:16] = int(n).to_bytes(8, "little", signed=True)
    buf[16:24] = int(d).to_bytes(8, "little", signed=True)
    if seed is not None:
        buf[24:32] = int(seed).to_bytes(8, "little", signed=True)
    return bytes(buf)


def _meta_rows(
    ids: list[str], meta: list[dict] | None
) -> Iterator[tuple[int, str, str | None]]:
    """Stream (rowid, record_id, json_meta) tuples without materialising the list."""
    n = len(ids)
    if meta is None:
        for i in range(n):
            yield (i, ids[i], None)
    else:
        for i in range(n):
            yield (i, ids[i], json.dumps(meta[i]))


class Corpus:
    """Paired bit-vector index + SQLite metadata store."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: np.dtype | type | None = None,
    ):
        """Open an existing corpus directory.

        Parameters
        ----------
        path : str | Path
            Directory containing ``index.bin`` and ``meta.db``.
        dtype : numpy dtype, optional
            Working precision for the reconstructed quantizer (used to
            encode queries at search time). Defaults to the
            :class:`SignBitQuantizer` default. Pass ``np.float64`` to
            match queries against corpora that were built before the
            f32-default change if you want bit-exact query encoding;
            recall is statistically identical either way.
        """
        self._dir = Path(path)
        bin_path = self._dir / _BIN_NAME
        db_path = self._dir / _DB_NAME

        if not bin_path.exists():
            raise FileNotFoundError(f"index.bin not found in {self._dir}")
        if not db_path.exists():
            raise FileNotFoundError(f"meta.db not found in {self._dir}")

        # Bound the file size before reading: refuse hostile/runaway files.
        bin_size = bin_path.stat().st_size
        if bin_size > (1 << 40):  # 1 TiB
            raise ValueError(
                f"index.bin is {bin_size} bytes (>1 TiB); refusing to load"
            )

        raw = np.fromfile(bin_path, dtype=np.uint8)
        n, d, seed, payload_off = _read_header(raw)

        # Verify file is exactly the size the header claims.
        codes_bytes = n * (d // 8)
        expected = payload_off + codes_bytes
        if raw.size < expected:
            raise ValueError(
                f"corrupt index: header claims {n}x{d // 8} bytes of codes "
                f"({codes_bytes} bytes payload) but file has only "
                f"{raw.size - payload_off} payload bytes"
            )
        if raw.size > expected:
            warnings.warn(
                f"index.bin has {raw.size - expected} trailing bytes after "
                f"payload; ignoring",
                UserWarning,
                stacklevel=2,
            )

        codes_flat = raw[payload_off : payload_off + codes_bytes]
        self._codes = codes_flat.reshape(n, d // 8)
        q_kwargs: dict = {"d": d, "seed": seed}
        if dtype is not None:
            q_kwargs["dtype"] = dtype
        self._quantizer = SignBitQuantizer(**q_kwargs)
        self._db_path = str(db_path)

        # Load corpus mean if persisted by build(center=True).
        mean_path = self._dir / _MEAN_NAME
        if mean_path.exists():
            self._mean: np.ndarray | None = np.load(mean_path)
            if self._mean.shape != (d,):
                raise ValueError(
                    f"mean.npy shape {self._mean.shape} does not match d={d}"
                )
        else:
            self._mean = None

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
        dtype: np.dtype | type | None = None,
    ) -> "Corpus":
        """Build a corpus from raw vectors and record IDs.

        ``dtype`` controls the quantizer's working precision; ``None``
        uses the :class:`SignBitQuantizer` default. Centering, when
        enabled, is computed in the input dtype before encoding.
        """
        vectors = np.asarray(vectors)
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
        # mode=0o700: corpus may contain sensitive embeddings + identifiers.
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(dest, 0o700)
        except OSError:
            pass

        q_kwargs: dict = {"d": d, "seed": seed}
        if dtype is not None:
            q_kwargs["dtype"] = dtype
        q = SignBitQuantizer(**q_kwargs)
        if center:
            corpus_mean = vectors.mean(axis=0)
            enc_vectors = vectors - corpus_mean
        else:
            corpus_mean = None
            enc_vectors = vectors
        codes = q.encode(enc_vectors)  # (n, d//8) uint8

        bin_path = dest / _BIN_NAME
        header = _write_header(n, d, seed)
        with open(bin_path, "wb") as f:
            f.write(header)
            f.write(codes.tobytes())
        try:
            os.chmod(bin_path, 0o600)
        except OSError:
            pass

        # Persist the corpus mean so search() can auto-center queries.
        # Mean is saved in input dtype so the round-trip is bit-exact for
        # callers that built with f32; the centering subtract at search
        # time is a single vector op, not a hot loop.
        mean_path = dest / _MEAN_NAME
        if corpus_mean is not None:
            np.save(mean_path, np.ascontiguousarray(corpus_mean))
            try:
                os.chmod(mean_path, 0o600)
            except OSError:
                pass
        else:
            mean_path.unlink(missing_ok=True)

        db_path = dest / _DB_NAME
        # missing_ok=True avoids a TOCTOU race with exists()/unlink().
        db_path.unlink(missing_ok=True)
        con = sqlite3.connect(str(db_path))
        try:
            con.executescript(_SCHEMA)
            # Generator avoids materialising n metadata rows in memory.
            con.executemany(
                "INSERT INTO corpus_meta (rowid, record_id, meta) VALUES (?, ?, ?)",
                _meta_rows(ids, meta),
            )
            con.commit()
        finally:
            con.close()
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass

        return cls(dest)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(self, query: np.ndarray, k: int = 10) -> list[Result]:
        """Return the top-k nearest neighbours with resolved metadata.

        When the corpus was built with ``center=True``, the stored corpus
        mean is automatically subtracted from the query before encoding.
        Callers do **not** need to center queries manually.
        """
        if k <= 0:
            return []

        query = np.asarray(query, dtype=self._quantizer.dtype)
        if self._mean is not None:
            query = query - self._mean

        indices, distances = self._quantizer.search(
            query, self._codes, k=k, return_distances=True
        )
        indices = np.asarray(indices)
        distances = np.asarray(distances)

        positions = indices.tolist()
        if not positions:
            return []

        lookup = self._fetch_meta(positions)

        results: list[Result] = []
        for rank, (pos, dist) in enumerate(zip(positions, distances.tolist())):
            record_id, meta = lookup.get(pos, (str(pos), None))
            results.append(
                Result(rank=rank, distance=int(dist), record_id=record_id, meta=meta)
            )
        return results

    def _fetch_meta(
        self, positions: Iterable[int]
    ) -> dict[int, tuple[str, Any]]:
        """Resolve positions to (record_id, meta), chunking under SQLite param limit."""
        positions = list(positions)
        out: dict[int, tuple[str, Any]] = {}
        if not positions:
            return out

        # mode=rw: refuse silent creation of a fresh DB if meta.db has been
        # deleted underneath us (CWE-367).
        uri = f"file:{self._db_path}?mode=rw"
        con = sqlite3.connect(uri, uri=True)
        try:
            for start in range(0, len(positions), _SQLITE_IN_CHUNK):
                chunk = positions[start : start + _SQLITE_IN_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = con.execute(
                    f"SELECT rowid, record_id, meta FROM corpus_meta "
                    f"WHERE rowid IN ({placeholders})",
                    chunk,
                ).fetchall()
                for rowid, record_id, meta_json in rows:
                    out[rowid] = (
                        record_id,
                        json.loads(meta_json) if meta_json else None,
                    )
        finally:
            con.close()
        return out

    def lookup(self, record_id: str) -> int | None:
        """Reverse lookup: external record ID → array position."""
        uri = f"file:{self._db_path}?mode=rw"
        con = sqlite3.connect(uri, uri=True)
        try:
            row = con.execute(
                "SELECT rowid FROM corpus_meta WHERE record_id = ? LIMIT 1",
                (record_id,),
            ).fetchone()
        finally:
            con.close()
        return int(row[0]) if row is not None else None

    @property
    def n(self) -> int:
        return self._codes.shape[0]

    @property
    def d(self) -> int:
        return self._quantizer.d

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    @property
    def mean(self) -> np.ndarray | None:
        """Corpus mean vector, or None if the corpus was not centered.

        Returns a copy to prevent accidental mutation.
        """
        return self._mean.copy() if self._mean is not None else None

    @property
    def centered(self) -> bool:
        """Whether this corpus was built with ``center=True``."""
        return self._mean is not None

    def __repr__(self) -> str:
        c = ", centered" if self.centered else ""
        return f"Corpus(n={self.n}, d={self.d}{c}, path={str(self._dir)!r})"
