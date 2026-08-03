"""remax._native — hardware-accelerated Hamming distance scan.

Compiles a tiny C library at first import (requires ``gcc`` or ``cc``),
caches it in a *user-private* cache directory, and loads it via
:mod:`ctypes`. Falls back gracefully: if compilation fails, ``AVAILABLE``
is ``False`` and callers should use the NumPy LUT path.

Zero extra dependencies — ctypes and subprocess are stdlib.

Security
--------
Compiled libraries are cached under ``~/.cache/remax`` (or ``$XDG_CACHE_HOME``
when set), which is owned and writable only by the current user. Older
versions wrote to ``$TMPDIR/remax_native``, a world-writable location on
most systems — a co-located unprivileged user could pre-place a malicious
``.so`` there with the predictable hash filename and have it loaded by
the next ``import remax`` (CWE-379). The ``REMAX_CACHE_DIR`` override is
preserved for build systems that pin a specific location, but defaults
are now safe.

Compilation writes to a temp file in the same directory and atomically
renames into place via ``os.replace`` so concurrent imports cannot load
a partially written library (CWE-367).

Performance
-----------
On x86-64 with hardware ``POPCNT`` (any CPU from ~2008 onward), the native
scan runs **roughly 25-35x faster than the NumPy LUT fallback**, at 5-11 GB/s
effective throughput (index bytes scanned / elapsed).

It is not a single number, and quoting it as one is what produced the two
mutually exclusive figures this docstring and README.md used to carry
("50-60x" here, "23x / 9.7 GB/s" there). The ratio moves with both ``n`` and
``d``, because the two paths have different memory behaviour: the NumPy path
materialises an ``(n, B)`` uint16 gather -- 2 bytes of intermediate per input
byte, so it touches ~3x the index and leaves cache early -- while the native
path streams the index once and writes 4 bytes per row.

Measured on an Intel Xeon @ 2.80 GHz, numpy 2.4, min of 9 runs
(``bench/native_speedup.py``):

    ==========  ========  ========  ============
    n           d=768     d=256     GB/s (d=768)
    ==========  ========  ========  ============
    10,000      32.9x     24.9x     10.8
    100,000     34.9x     32.2x     10.0
    1,000,000   33.5x     27.7x      6.8
    ==========  ========  ========  ============

Throughput falls off between 100k and 1M because that is where the index
stops fitting in last-level cache; past that point the scan is bandwidth-
bound, which is the regime the design targets.

The old "within a factor of 2 of raw memcpy bandwidth" line is gone rather
than restated. It compared unlike traffic: memcpy moves ``2*n*B`` bytes
(read and write) where this scan reads ``n*B`` and writes ``4n``, so the
comparison can be made to say almost anything. ``bench/native_speedup.py``
still prints a memcpy column, labelled as a scale reference, not headroom.

Reproduce with ``python3 bench/native_speedup.py [--d D] [--sizes N ...]``.
These are hardware-specific; re-measure before quoting them elsewhere.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["AVAILABLE", "hamming_distances_native"]

logger = logging.getLogger(__name__)

# ── C source ──────────────────────────────────────────────────────────

_C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*
 * Hamming distance from query (B bytes) to each of n corpus rows.
 * Uses hardware popcount via __builtin_popcountll (gcc/clang).
 * Processes 8 bytes (64 bits) per iteration for maximum throughput.
 *
 * n and B are int64_t so corpora with > 2^31 rows are handled correctly.
 * (32-bit `int` truncated silently and produced garbage at scale.)
 *
 * out is int32_t: a Hamming distance is at most 8*B bits, which would
 * overflow int32 only past a ~256 MB code — never in practice — so the
 * narrower output halves this bandwidth-bound scan's write traffic and lets
 * the downstream top-k argpartition run over int32. The accumulator stays
 * int64 and is cast on store.
 */
void hamming_scan(
    const uint8_t *corpus,   /* (n, B) row-major */
    const uint8_t *query,    /* (B,) */
    int32_t       *out,      /* (n,) output distances */
    int64_t        n,        /* number of corpus rows */
    int64_t        B         /* bytes per code */
) {
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *row = corpus + (size_t)i * (size_t)B;
        int64_t dist = 0;
        int64_t j = 0;

        /* Main loop: 8 bytes at a time with 64-bit popcount */
        for (; j + 8 <= B; j += 8) {
            uint64_t a, b;
            memcpy(&a, row + j, 8);
            memcpy(&b, query + j, 8);
            dist += __builtin_popcountll(a ^ b);
        }

        /* Tail: remaining bytes */
        for (; j < B; j++) {
            dist += __builtin_popcount(row[j] ^ query[j]);
        }

        out[i] = (int32_t)dist;
    }
}
"""

# Source hash determines cache validity — recompile only when C changes.
_SOURCE_HASH = hashlib.sha256(_C_SOURCE.encode()).hexdigest()[:16]


# ── Compilation ───────────────────────────────────────────────────────

def _cache_dir() -> Path:
    """User-private cache directory for compiled libraries.

    Resolution order:
      1. ``REMAX_CACHE_DIR`` (explicit override, e.g., for CI/Nix)
      2. ``$XDG_CACHE_HOME/remax`` (XDG Base Directory spec)
      3. ``~/.cache/remax`` (POSIX default)
      4. ``$TMPDIR/remax_native_<uid>`` (fallback when home is unwritable;
         UID-suffixed to keep the dir per-user even on shared systems)

    Older versions used ``$TMPDIR/remax_native`` (world-writable), which
    let a co-located attacker pre-place a malicious ``.so`` for loading
    by ``ctypes.CDLL``. See CWE-379.
    """
    base = os.environ.get("REMAX_CACHE_DIR")
    if base:
        d = Path(base)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            d = Path(xdg) / "remax"
        else:
            try:
                home = Path.home()
                d = home / ".cache" / "remax"
            except (RuntimeError, KeyError):
                uid = os.getuid() if hasattr(os, "getuid") else 0
                d = Path(tempfile.gettempdir()) / f"remax_native_{uid}"
                logger.warning(
                    "remax: could not resolve home directory; using %s. "
                    "Set REMAX_CACHE_DIR or HOME for a private cache.",
                    d,
                )
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _lib_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def _compile() -> Path | None:
    """Compile the C source to a shared library. Returns path or None.

    Writes to a unique temporary file, then atomically renames into the
    final cache path. Concurrent compilers each write their own temp file;
    the last ``os.replace`` wins. Loaders never observe a partially
    written ``.so`` (CWE-367).
    """
    cache = _cache_dir()
    suffix = _lib_suffix()
    lib_name = f"remax_hamming_{_SOURCE_HASH}{suffix}"
    lib_path = cache / lib_name

    if lib_path.exists():
        return lib_path

    src_path = cache / f"remax_hamming_{_SOURCE_HASH}.c"
    src_path.write_text(_C_SOURCE)

    for compiler in ("gcc", "cc"):
        tmp_path = cache / f"{lib_name}.{os.getpid()}.tmp"
        cmd = [
            compiler, "-shared", "-fPIC", "-O3",
            "-o", str(tmp_path), str(src_path),
        ]
        if platform.machine() in ("x86_64", "AMD64"):
            cmd.insert(4, "-mpopcnt")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tmp_path.unlink(missing_ok=True)
            continue
        except OSError as e:
            logger.info("remax: compiler %s unusable (%s)", compiler, e)
            tmp_path.unlink(missing_ok=True)
            continue

        if result.returncode == 0 and tmp_path.exists():
            try:
                os.replace(str(tmp_path), str(lib_path))
                try:
                    os.chmod(lib_path, 0o600)
                except OSError:
                    pass
                logger.debug("remax: compiled native scan with %s", compiler)
                return lib_path
            except OSError as e:
                logger.info("remax: failed to publish compiled lib: %s", e)
                tmp_path.unlink(missing_ok=True)
                continue

        logger.debug(
            "remax: %s exited with returncode=%d; stderr=%r",
            compiler, result.returncode, (result.stderr or "")[:1024],
        )
        tmp_path.unlink(missing_ok=True)

    logger.info(
        "remax: native scan compilation failed (no working gcc/cc?); "
        "falling back to NumPy LUT."
    )
    return None


# ── ctypes wrapper ────────────────────────────────────────────────────

def _load_lib(path: Path) -> ctypes.CDLL | None:
    """Load the compiled library and set argtypes."""
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        logger.info("remax: failed to load native lib: %s", e)
        return None

    lib.hamming_scan.restype = None
    lib.hamming_scan.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    return lib


_lib_path = _compile()
_lib = _load_lib(_lib_path) if _lib_path else None

AVAILABLE: bool = _lib is not None


def hamming_distances_native(
    codes: np.ndarray, query_code: np.ndarray, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Hamming distance from query_code to every row of codes.

    ``out``, when given, is an ``(n,)`` int32 C-contiguous buffer written in
    place and returned. The kernel assigns every element (``out[i] = dist``),
    so nothing from a previous call survives.

    This function no longer copies ``codes``. It used to open with
    ``np.ascontiguousarray(codes, dtype=np.uint8)`` — the third such call on
    a path that already had two, and the one in the worst position, because
    it sits inside the per-query loop. A non-contiguous array is now a
    ``ValueError`` here rather than a silent whole-index copy: the public
    entry point (:func:`remax.packing.as_codes`) is where a copy may happen,
    once, with a warning. Passing a raw pointer to a strided buffer would
    read garbage, so this check is load-bearing, not defensive.
    """
    if _lib is None:
        raise RuntimeError(
            "Native scan not available; check remax._native.AVAILABLE "
            "before calling."
        )

    query_code = np.ascontiguousarray(query_code, dtype=np.uint8)

    if not isinstance(codes, np.ndarray):
        raise ValueError(f"codes must be a numpy array, got {type(codes).__name__}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got ndim={codes.ndim}")
    if codes.dtype != np.uint8:
        raise ValueError(f"codes must be uint8, got dtype={codes.dtype}")
    if not codes.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "codes must be C-contiguous; the native kernel reads a raw "
            "pointer and a strided buffer would give wrong distances. Pass "
            "it through remax.packing.as_codes (or np.ascontiguousarray) "
            "first — once, outside your query loop."
        )
    if query_code.ndim != 1:
        raise ValueError(f"query_code must be 1-D, got ndim={query_code.ndim}")
    if query_code.shape[0] != codes.shape[1]:
        raise ValueError(
            f"query_code length {query_code.shape[0]} does not match "
            f"codes row length {codes.shape[1]}"
        )

    n, B = codes.shape
    if out is None:
        out = np.empty(n, dtype=np.int32)
    else:
        if out.shape != (n,):
            raise ValueError(f"out has shape {out.shape}; expected ({n},)")
        if out.dtype != np.int32:
            raise ValueError(f"out must be int32, got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        if not out.flags["WRITEABLE"]:
            raise ValueError("out must be writeable")

    _lib.hamming_scan(
        codes.ctypes.data,
        query_code.ctypes.data,
        out.ctypes.data,
        ctypes.c_int64(n),
        ctypes.c_int64(B),
    )
    return out
