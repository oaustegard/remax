"""remax._native — hardware-accelerated Hamming distance scan.

Compiles a tiny C library at first import (requires ``gcc`` or ``cc``),
caches it in a platform-appropriate temp directory, and loads it via
:mod:`ctypes`. Falls back gracefully: if compilation fails, ``AVAILABLE``
is ``False`` and callers should use the NumPy LUT path.

Zero extra dependencies — ctypes and subprocess are stdlib.

Performance
-----------
On x86-64 with hardware ``POPCNT`` (any CPU from ~2008 onward), the native
scan achieves ~10 GB/s effective throughput — roughly 50–60× faster than
the NumPy LUT path, and within a factor of 2 of raw ``memcpy`` bandwidth.
The scan is memory-bound at native speed; further optimisation requires
algorithmic changes (coarse indexing), not faster popcount.
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
#include <string.h>

/*
 * Hamming distance from query (B bytes) to each of n corpus rows.
 * Uses hardware popcount via __builtin_popcountll (gcc/clang).
 * Processes 8 bytes (64 bits) per iteration for maximum throughput.
 */
void hamming_scan(
    const uint8_t *corpus,   /* (n, B) row-major */
    const uint8_t *query,    /* (B,) */
    int64_t       *out,      /* (n,) output distances */
    int            n,        /* number of corpus rows */
    int            B         /* bytes per code */
) {
    for (int i = 0; i < n; i++) {
        const uint8_t *row = corpus + (size_t)i * B;
        int64_t dist = 0;
        int j = 0;

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

        out[i] = dist;
    }
}
"""

# Source hash determines cache validity — recompile only when C changes.
_SOURCE_HASH = hashlib.md5(_C_SOURCE.encode()).hexdigest()[:12]


# ── Compilation ───────────────────────────────────────────────────────

def _cache_dir() -> Path:
    """Platform-appropriate cache directory for compiled libraries."""
    base = os.environ.get("REMAX_CACHE_DIR")
    if base:
        d = Path(base)
    else:
        d = Path(tempfile.gettempdir()) / "remax_native"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lib_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def _compile() -> Path | None:
    """Compile the C source to a shared library. Returns path or None."""
    suffix = _lib_suffix()
    lib_name = f"remax_hamming_{_SOURCE_HASH}{suffix}"
    lib_path = _cache_dir() / lib_name

    # Already compiled and cached?
    if lib_path.exists():
        return lib_path

    src_path = _cache_dir() / f"remax_hamming_{_SOURCE_HASH}.c"
    src_path.write_text(_C_SOURCE)

    # Try gcc first, fall back to cc.
    for compiler in ("gcc", "cc"):
        cmd = [
            compiler, "-shared", "-fPIC", "-O3",
            "-o", str(lib_path), str(src_path),
        ]

        # Add -mpopcnt on x86_64 (noop on ARM which has it always).
        if platform.machine() in ("x86_64", "AMD64"):
            cmd.insert(4, "-mpopcnt")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and lib_path.exists():
                logger.debug("remax: compiled native scan with %s", compiler)
                return lib_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    logger.info(
        "remax: native scan compilation failed (no gcc/cc?); "
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
        ctypes.c_void_p,   # corpus
        ctypes.c_void_p,   # query
        ctypes.c_void_p,   # out
        ctypes.c_int,      # n
        ctypes.c_int,      # B
    ]
    return lib


# ── Module-level init ─────────────────────────────────────────────────

_lib_path = _compile()
_lib = _load_lib(_lib_path) if _lib_path else None

AVAILABLE: bool = _lib is not None
"""True if the native scan kernel compiled and loaded successfully."""


def hamming_distances_native(
    codes: np.ndarray, query_code: np.ndarray
) -> np.ndarray:
    """Hamming distance from query_code to every row of codes.

    Drop-in replacement for :func:`remax.packing.hamming_distances`
    when :data:`AVAILABLE` is True.

    Parameters
    ----------
    codes : np.ndarray, shape (n, B), dtype uint8, C-contiguous
    query_code : np.ndarray, shape (B,), dtype uint8, C-contiguous

    Returns
    -------
    distances : np.ndarray, shape (n,), dtype int64
    """
    if _lib is None:
        raise RuntimeError(
            "Native scan not available; check remax._native.AVAILABLE "
            "before calling."
        )

    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    query_code = np.ascontiguousarray(query_code, dtype=np.uint8)

    n, B = codes.shape
    out = np.empty(n, dtype=np.int64)

    _lib.hamming_scan(
        codes.ctypes.data,
        query_code.ctypes.data,
        out.ctypes.data,
        n,
        B,
    )
    return out
