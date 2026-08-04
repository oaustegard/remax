"""remax.packing — bit-pack utilities and Hamming-distance scan.

Three primitives:

* :func:`encode_signs` — already-rotated floats → bit-packed ``uint8`` codes.
* :func:`hamming_distances` — broadcast XOR + popcount-LUT sum over a corpus.

When a C compiler is available, ``hamming_distances`` dispatches to a native
kernel using hardware ``POPCNT`` (~25–35× faster than the NumPy LUT fallback;
the ratio depends on ``n`` and ``d`` — :mod:`remax._native` carries the table).
The native path compiles automatically at first import and is cached; no extra
dependencies are required.  See :mod:`remax._native` for details.

Parallelism
-----------
The C kernel is called through :mod:`ctypes`, which **releases the GIL** for
the duration of the call. So the scan parallelises across Python threads with
no change to the C at all: :func:`hamming_distances` splits ``codes`` into row
blocks and dispatches one kernel call per block into a shared
:class:`~concurrent.futures.ThreadPoolExecutor`, each writing into its own
slice of one preallocated output. Row blocks partition the corpus, so the
result is bit-identical to the serial scan by construction, not by tolerance.

Threading is **off by default** (``threads=None`` → :func:`get_default_threads`
→ 1) because a library that silently spawns threads inside somebody else's
worker pool is a bad neighbour. Opt in per call (``threads=4``,
``threads="auto"``), process-wide (:func:`set_default_threads`), or by
environment (``REMAX_THREADS=auto``).

Removed, 2026-08-03: ``hamming_search`` and ``asymmetric_search``
-----------------------------------------------------------------
Two single-query top-k wrappers used to live here and were exported from
``remax``. Nothing in the library ever called them: every search path
(:meth:`SignBitQuantizer.search`, ``SignBitQuantizer.search_asymmetric``, and
the stacked equivalents) composes ``hamming_distances`` / ``asymmetric_scores``
with ``stable_top_k`` directly, because it needs the batch loop and the
reusable output buffer the wrappers did not provide.

They were also the more dangerous of the two spellings: both took an
**already-rotated** query, so a caller passing a raw vector got plausible wrong
answers rather than an error, and neither accepted a batch. Use the quantizer
methods. If you genuinely want the functional form, it is one line each::

    order = stable_top_k(hamming_distances(codes, encode_signs(q @ R)), k)
    order = stable_top_k(-asymmetric_scores(q @ R, codes), k)
"""

from __future__ import annotations

import atexit
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _native

__all__ = [
    "POPCOUNT_LUT",
    "NonContiguousCodesWarning",
    "as_codes",
    "asymmetric_scores",
    "asymmetric_tables",
    "counting_top_k",
    "encode_signs",
    "get_default_threads",
    "hamming_distances",
    "hamming_topk_batch",
    "resolve_threads",
    "scores_from_table",
    "set_default_threads",
    "stable_top_k",
]


# ── thread-count knob ────────────────────────────────────────────────────────
#
# Default 1: today's behaviour, exactly. A caller who has not asked for threads
# gets the same single-threaded call sequence, the same peak memory, and the
# same lack of a background pool as before this existed.

_MIN_BYTES_PER_THREAD = 2 << 20
"""Below this many *bytes* of code per thread, run serially instead.

Bytes, not rows. The scan is bandwidth-bound, so what a thread has to do is
set by how many bytes its block holds; a row count means different work at
B=32 and B=96 and would put the cutoff in a different place for each.

Derived from measurement on the box in ``bench/results/QUERY_PATH_SPEED.md``,
where pool dispatch costs ~60 us for 2-4 tasks (~167 us at 8, oversubscribed)
against a ~9.7 GB/s single-core scan:

    bytes/thread   T=4 speedup
    0.8 MB         0.76x   (net loss)
    1.6 MB         1.48x
    3.2 MB         2.08x
    6.4 MB         3.27x

Break-even is around 1 MB per thread; the floor sits at 2 MB so the threaded
path is only taken where it was measured to win with margin.

This started at 16384 *rows*, which at B=32 is 512 KB — comfortably inside the
losing region. The benchmark caught it: threading at n=1e5 measured 0.5-0.8x,
a slowdown that the correctness gate is structurally unable to see, because the
answers were right the whole time.
"""


def _max_workers_for(nbytes: int) -> int:
    """How many threads this much code can keep usefully busy."""
    return max(1, nbytes // _MIN_BYTES_PER_THREAD)

_default_threads: int = 1
_pool_lock = threading.Lock()
_pools: dict[int, ThreadPoolExecutor] = {}


def _cpu_count() -> int:
    try:
        # Respects sched_setaffinity / cgroup pinning where available.
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def _parse_threads(value, *, where: str) -> int:
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError(
                f"{where} must be a positive int or 'auto', got {value!r}"
            )
        return max(1, _cpu_count())
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{where} must be a positive int or 'auto', got "
            f"{type(value).__name__}"
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{where} must be >= 1 (or 'auto'), got {value}")
    return value


def set_default_threads(threads: int | str) -> int:
    """Set the process-wide default thread count for the scan. Returns it.

    ``"auto"`` resolves to the number of CPUs this process may run on.
    """
    global _default_threads
    _default_threads = _parse_threads(threads, where="threads")
    return _default_threads


def get_default_threads() -> int:
    """The thread count used when a call passes ``threads=None``."""
    return _default_threads


def resolve_threads(threads: int | str | None) -> int:
    """Resolve a ``threads=`` argument to a concrete positive int."""
    if threads is None:
        return _default_threads
    return _parse_threads(threads, where="threads")


def _env_default() -> None:
    raw = os.environ.get("REMAX_THREADS")
    if not raw:
        return
    try:
        set_default_threads(raw if raw.lower() == "auto" else int(raw))
    except (TypeError, ValueError):
        warnings.warn(
            f"ignoring REMAX_THREADS={raw!r}: expected a positive int or "
            f"'auto'",
            UserWarning,
            stacklevel=2,
        )


_env_default()


def _pool(workers: int) -> ThreadPoolExecutor:
    """A process-wide pool per worker count, created on first use.

    Reused rather than created per call: constructing a pool costs a thread
    spawn per worker, which at a few hundred microseconds would swamp the
    milliseconds of scan it is meant to accelerate.
    """
    with _pool_lock:
        pool = _pools.get(workers)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="remax-scan"
            )
            _pools[workers] = pool
        return pool


def _shutdown_pools() -> None:  # pragma: no cover - interpreter teardown
    with _pool_lock:
        pools = list(_pools.values())
        _pools.clear()
    for p in pools:
        p.shutdown(wait=False)


def _reset_pools_after_fork() -> None:  # pragma: no cover - fork child
    """A forked child inherits pool objects whose threads did not survive.

    The lock is rebuilt too: if some other thread held it at ``fork()`` time it
    is inherited locked, and nothing in the child will ever release it.
    """
    global _pool_lock
    _pool_lock = threading.Lock()
    _pools.clear()


atexit.register(_shutdown_pools)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pools_after_fork)


class NonContiguousCodesWarning(UserWarning):
    """A code matrix had to be copied to be scanned.

    Its own class so callers can escalate it (``warnings.simplefilter("error",
    NonContiguousCodesWarning)``) or silence it deliberately, without touching
    every other warning remax might raise.
    """


def as_codes(codes: np.ndarray, *, argname: str = "codes") -> np.ndarray:
    """Validate a packed code matrix; copy only when unavoidable, and say so.

    The single place a code matrix is checked. There used to be three —
    ``core.SignBitQuantizer.search``, :func:`hamming_distances` and
    ``_native.hamming_distances_native`` each opened with
    ``np.ascontiguousarray(codes, dtype=np.uint8)``. On a well-formed index all
    three are no-ops, so the redundancy cost nothing and looked harmless. On a
    *strided view* it was three chances to copy the entire index, silently, on
    a path that runs once per query — and a strided view is not an exotic
    input. ``corpus.codes[::2]``, a subsampled evaluation set, a slice along
    the wrong axis: all produce one.

    Measured at 1M x 256 bits (32 MB): 5.6 ms contiguous, 16.6 ms through a
    strided view, every millisecond of the difference invisible.

    So: exactly one site may copy, and when it does it warns with the size.

    Raises
    ------
    ValueError
        If ``codes`` is not 2-D, or its dtype is not ``uint8``. A float or
        int64 array here is a caller error — packed codes are bytes — and the
        old silent ``dtype=np.uint8`` cast turned a wrong-array bug into wrong
        distances rather than an exception.

    Warns
    -----
    NonContiguousCodesWarning
        If a copy was required. The array is still copied and the call still
        succeeds; the point is that it stops being invisible.
    """
    arr = np.asarray(codes)
    if arr.ndim != 2:
        raise ValueError(f"{argname} must be 2-D, got ndim={arr.ndim}")
    if arr.dtype != np.uint8:
        raise ValueError(
            f"{argname} must be uint8 (packed bits), got dtype={arr.dtype}. "
            f"Encode with remax.encode_signs, or cast explicitly if you are "
            f"sure this array already holds packed bytes."
        )
    if not arr.flags["C_CONTIGUOUS"]:
        warnings.warn(
            f"{argname} is not C-contiguous; copying "
            f"{arr.nbytes / 1e6:.1f} MB to scan it. This happens on every "
            f"call. Hoist the copy with np.ascontiguousarray(codes) once, "
            f"outside your query loop.",
            NonContiguousCodesWarning,
            stacklevel=3,
        )
        arr = np.ascontiguousarray(arr)
    return arr


def _as_out(out: np.ndarray | None, n: int) -> np.ndarray:
    """Validate a caller-supplied output buffer, or allocate one."""
    if out is None:
        return np.empty(n, dtype=np.int32)
    if not isinstance(out, np.ndarray):
        raise TypeError(f"out must be a numpy array, got {type(out).__name__}")
    if out.shape != (n,):
        raise ValueError(
            f"out has shape {out.shape}; expected ({n},) to match the corpus"
        )
    if out.dtype != np.int32:
        raise ValueError(
            f"out must be int32 (the distance dtype), got {out.dtype}"
        )
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")
    if not out.flags["WRITEABLE"]:
        raise ValueError("out must be writeable")
    return out

# 256-entry byte-popcount lookup. uint16 is plenty (max value 8 per byte).
POPCOUNT_LUT: np.ndarray = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint16
)


def encode_signs(X_rotated: np.ndarray) -> np.ndarray:
    """Pack the sign bits of an already-rotated array into ``uint8`` bytes.

    Convention: ``x > 0`` → bit ``1``; ``x ≤ 0`` → bit ``0``. Bits are
    packed big-endian within each byte (numpy's default).

    Parameters
    ----------
    X_rotated : np.ndarray, shape (n, d) or (d,)
        Already-rotated input. Trailing dim ``d`` must be divisible by 8.

    Returns
    -------
    codes : np.ndarray, dtype uint8
        ``(n, d // 8)`` if input is 2-D; ``(d // 8,)`` if input is 1-D.
    """
    X = np.asarray(X_rotated)
    squeezed = False
    if X.ndim == 1:
        X = X[None, :]
        squeezed = True
    elif X.ndim != 2:
        raise ValueError(
            f"X_rotated must be 1-D or 2-D, got ndim={X.ndim}"
        )
    if X.shape[-1] % 8 != 0:
        raise ValueError(
            f"trailing dim must be divisible by 8 (got {X.shape[-1]}); "
            "remax codes are bit-packed into uint8 bytes."
        )
    bits = X > 0
    codes = np.packbits(bits, axis=-1)
    return codes[0] if squeezed else codes


def _scan_serial(codes: np.ndarray, q: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Scan every row of ``codes`` into ``out``. One thread, no dispatch.

    The single place the kernel choice (native vs NumPy LUT) is made, so the
    threaded path and the serial path cannot drift apart.
    """
    if _native.AVAILABLE:
        return _native.hamming_distances_native(codes, q, out=out)
    xor = np.bitwise_xor(codes, q[None, :])
    # POPCOUNT_LUT[xor] is (n, B) uint16 — popcount per byte. Sum across
    # bytes gives total Hamming distance per row.
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32, out=out)


def _row_blocks(n: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``parts`` contiguous, non-overlapping blocks.

    Exhaustive and disjoint by construction: block ``i`` is
    ``[bounds[i], bounds[i+1])`` from a single monotone bound array, so no row
    can be visited twice and none can be skipped. A "split" written as
    ``n // parts`` per block silently drops the ``n % parts`` tail.
    """
    bounds = np.linspace(0, n, parts + 1).astype(np.int64)
    return [
        (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ]


def _scan_threaded(
    codes: np.ndarray, q: np.ndarray, out: np.ndarray, workers: int
) -> np.ndarray:
    """Row-block the scan across ``workers`` threads into one shared ``out``.

    Bit-identical to :func:`_scan_serial` by construction: the blocks
    partition the rows, each thread writes only its own disjoint slice, and
    the per-row computation is unchanged. There is no reduction, so there is
    no ordering to get wrong.
    """
    blocks = _row_blocks(codes.shape[0], workers)
    if len(blocks) <= 1:
        return _scan_serial(codes, q, out)
    pool = _pool(workers)
    futures = [
        pool.submit(_scan_serial, codes[a:b], q, out[a:b]) for a, b in blocks
    ]
    for f in futures:
        f.result()  # re-raises anything a worker hit
    return out


def hamming_distances(
    codes: np.ndarray,
    query_code: np.ndarray,
    *,
    out: np.ndarray | None = None,
    threads: int | str | None = None,
) -> np.ndarray:
    """Hamming distance from ``query_code`` to every row of ``codes``.

    Parameters
    ----------
    codes : np.ndarray, shape (n, B), dtype uint8
        Bit-packed corpus. Must be C-contiguous uint8; see :func:`as_codes`
        for what happens when it is not.
    query_code : np.ndarray, shape (B,), dtype uint8
        Bit-packed query.
    out : np.ndarray, shape (n,), dtype int32, keyword-only
        Destination buffer. When given, distances are written into it and it
        is returned, so a caller looping over queries allocates one ``(n,)``
        int32 array instead of one per query — at n=1M that is 4 MB of
        allocate-and-free per query. Omitting it preserves the previous
        behaviour exactly (a fresh array each call).

        The buffer is fully overwritten every call, so stale values from a
        previous query cannot survive into the next.
    threads : int | "auto" | None, keyword-only
        Threads to split the scan across. ``None`` (the default) takes the
        process-wide default from :func:`get_default_threads`, which is 1
        unless changed — so the default behaviour is exactly the previous
        single-threaded one. ``"auto"`` uses the available CPU count.

        The kernel is reached through ctypes, which releases the GIL, so this
        is real parallelism with no change to the C. Row blocks partition the
        corpus and each thread writes a disjoint slice of ``out``, so the
        result is **bit-identical** to a serial scan for every thread count —
        this is a property of the decomposition, not a tolerance.

        Small corpora bypass the pool entirely (see
        ``_MIN_BYTES_PER_THREAD``): dispatch overhead exceeds the work, and
        threading them measures *slower*.

    Returns
    -------
    distances : np.ndarray, shape (n,), dtype int32
        Per-row Hamming distance, in ``[0, 8 * B]``. int32 is exact for any
        practical code width (8 * B overflows int32 only past a ~256 MB code)
        and keeps the downstream top-k argpartition narrow; ``search`` widens
        the returned top-k slice back to int64 for its public contract.
    """
    codes = as_codes(codes)
    q = np.ascontiguousarray(query_code, dtype=np.uint8)
    if q.ndim != 1 or q.shape[0] != codes.shape[1]:
        raise ValueError(
            f"query_code shape {q.shape} incompatible with "
            f"codes shape {codes.shape}"
        )
    n = codes.shape[0]
    out = _as_out(out, n)
    workers = resolve_threads(threads)
    # Cap the worker count so no thread gets a block too small to pay for its
    # own dispatch, and so a small corpus never touches the pool at all.
    workers = min(workers, _max_workers_for(codes.nbytes))
    if workers <= 1:
        return _scan_serial(codes, q, out)
    return _scan_threaded(codes, q, out, workers)


# ── counting (histogram) select ──────────────────────────────────────────────
#
# A Hamming distance over a B-byte code is an integer in [0, 8B]: 257 possible
# values at B=32, 2049 at B=256. Selection over a bounded integer alphabet does
# not need a comparison-based partition at all — one histogram pass finds the
# cutoff value exactly, and a second pass collects the members.
#
# The motivation is allocation, not asymptotics. Both are O(n), but
# np.argpartition materialises a full (n,) intp permutation: 8n bytes, 80 MB at
# n=1e7, on a path whose whole point is not touching more memory than the index
# it is scanning. The histogram is 8*(8B+1) bytes — 2 KB at B=32 — and the
# collection allocates O(k) plus one bounded chunk.
#
# CONTRACT. stable_top_k documents byte-for-byte equivalence to
# np.argsort(dists, kind="stable")[:k], and remax PR #32 exists because a naive
# argpartition broke exactly that on tie-dense input. The counting select is
# held to the identical contract, and it gets there structurally:
#
#   * every index with distance < cutoff is taken (there are fewer than k of
#     them, by the definition of the cutoff), ordered by (distance, index);
#   * the remainder is filled from distance == cutoff in ASCENDING INDEX order,
#     which is what a stable sort does with a tie group.
#
# Taking the equal-group from anywhere but its front is the tie-order break
# this whole path has to not commit. bench/gates/query_path_gate.py simulates
# it (--simulate counting-select-tie-break) and it goes red.

#: Below this many rows, argpartition wins on *time* — the histogram's extra
#: linear passes cost more than the permutation they avoid. Measured on the box
#: in bench/results/QUERY_PATH_SPEED.md (min of 11, k=10 and k=100):
#:
#:     n        counting vs argpartition
#:     1e5      0.99-1.01x
#:     3e5      0.88-0.89x
#:     1e6      0.88-0.95x
#:     3e6      1.13-1.25x
#:     1e7      2.43-2.63x
#:
#: so the crossover is around 2e6 and the threshold sits at 2^21 = 2,097,152.
#: An earlier 2^20 put it inside the losing region — measured, not reasoned.
#:
#: Note what the threshold is NOT chosen on: the counting path allocates ~2 KB
#: against argpartition's 8n bytes at *every* size, and that advantage is
#: largest in relative terms exactly where it loses on time. Time is the
#: criterion here because remax has no evidence about the allocation mattering;
#: a caller under memory pressure can call counting_top_k directly.
COUNTING_MIN_N = 1 << 21

#: Widest value alphabet the histogram is allowed. 65537 int64 counters is
#: 512 KB — past that the histogram stops being the cheap side. B=8192 (a
#: 65536-bit code) is far beyond anything remax ships.
COUNTING_MAX_VALUE = 1 << 16

#: Rows per pass in the histogram and collection loops. Small enough that the
#: intp cast np.bincount performs internally stays in cache (measured 19 ms at
#: 2^18-2^20 vs 58 ms unchunked, n=1e7), large enough that the Python-level
#: loop is not the cost.
_COUNT_CHUNK = 1 << 20


class _NotCountable(Exception):
    """Internal: this input is not a bounded non-negative integer array."""


def _histogram(dists: np.ndarray, hi: int) -> np.ndarray:
    """Chunked value histogram over ``[0, hi]``.

    Chunked for two reasons, both measured. ``np.bincount`` casts its input to
    ``intp`` internally, so an unchunked call on an ``(n,) int32`` array
    allocates the very 8n bytes this path exists to avoid; and the chunked form
    is ~3x faster at n=1e7 besides, because the cast stays in cache.
    """
    counts = np.zeros(hi + 1, dtype=np.int64)
    for start in range(0, dists.shape[0], _COUNT_CHUNK):
        block = dists[start : start + _COUNT_CHUNK]
        try:
            part = np.bincount(block, minlength=hi + 1)
        except (ValueError, TypeError):  # negative or non-integer values
            raise _NotCountable from None
        if part.size != counts.size:  # a value above the declared bound
            raise _NotCountable
        counts += part
    return counts


def counting_top_k(
    dists: np.ndarray, k: int, *, value_bound: int | None = None
) -> np.ndarray:
    """Exact stable top-k over **non-negative integer** distances.

    Byte-for-byte identical to ``np.argsort(dists, kind="stable")[:k]``, same
    as :func:`stable_top_k` — this is the histogram implementation of the same
    contract, restricted to the dtype where a histogram is possible.

    Parameters
    ----------
    dists : np.ndarray, shape (n,)
        Non-negative integers. A float array raises ``TypeError``: this is the
        integer path, and silently rounding a float would change the answer.
    k : int
        Number of indices, clamped to ``min(k, n)``.
    value_bound : int, optional
        Largest value ``dists`` can take (``8 * B`` for a B-byte Hamming code).
        Supplying it skips an ``O(n)`` max reduction — ~3 ms at n=1e7 here. A
        bound that turns out to be too small is detected, not trusted: the
        histogram raises rather than dropping the rows above it.

    Raises
    ------
    TypeError
        ``dists`` is not an integer array.
    ValueError
        ``k`` is not positive, or the values are negative / wider than
        :data:`COUNTING_MAX_VALUE`.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    dists = np.asarray(dists)
    if not np.issubdtype(dists.dtype, np.integer):
        raise TypeError(
            f"counting_top_k needs an integer array, got {dists.dtype}; use "
            f"stable_top_k for float scores."
        )
    try:
        return _counting_top_k(dists, k, value_bound)
    except _NotCountable as exc:
        raise ValueError(
            "counting_top_k needs non-negative values no wider than "
            f"COUNTING_MAX_VALUE={COUNTING_MAX_VALUE}"
        ) from exc


def _counting_top_k(
    dists: np.ndarray, k: int, value_bound: int | None
) -> np.ndarray:
    """The histogram select proper. Raises :class:`_NotCountable` if it can't."""
    n = dists.shape[0]
    k_eff = min(k, n)
    if n == 0:
        return np.empty(0, dtype=np.intp)

    if value_bound is None:
        hi = int(dists.max())
        if hi < 0:
            raise _NotCountable
    else:
        hi = int(value_bound)
    if hi < 0 or hi >= COUNTING_MAX_VALUE:
        raise _NotCountable

    counts = _histogram(dists, hi)
    cum = np.cumsum(counts)
    # cutoff = smallest value v with (number of entries <= v) >= k_eff.
    cutoff = int(np.searchsorted(cum, k_eff, side="left"))
    n_below = int(cum[cutoff - 1]) if cutoff > 0 else 0
    n_equal = k_eff - n_below          # 0 < n_equal <= counts[cutoff]

    # Collect. Strictly-below indices number n_below < k_eff, so they fit in
    # O(k) memory; the equal group is truncated to the first n_equal hits,
    # which is what makes this stable and what bounds the allocation when a
    # tie group spans the whole corpus.
    below_idx: list[np.ndarray] = []
    below_val: list[np.ndarray] = []
    equal_idx: list[np.ndarray] = []
    got_below = 0
    got_equal = 0
    for start in range(0, n, _COUNT_CHUNK):
        block = dists[start : start + _COUNT_CHUNK]
        if cutoff > 0 and got_below < n_below:
            sel = np.flatnonzero(block < cutoff)
            if sel.size:
                below_idx.append(sel + start)
                below_val.append(block[sel])
                got_below += int(sel.size)
        if got_equal < n_equal:
            sel_eq = np.flatnonzero(block == cutoff)
            if sel_eq.size:
                take = sel_eq[: n_equal - got_equal]
                equal_idx.append(take + start)
                got_equal += int(take.size)
        if got_below >= n_below and got_equal >= n_equal:
            break

    parts: list[np.ndarray] = []
    if below_idx:
        idx = np.concatenate(below_idx)
        val = np.concatenate(below_val)
        # Chunks are visited in order and flatnonzero is ascending, so `idx`
        # is globally ascending; a stable sort on value therefore orders by
        # (distance, index) — the argsort(kind="stable") contract.
        parts.append(idx[np.argsort(val, kind="stable")])
    if equal_idx:
        parts.append(np.concatenate(equal_idx))
    if not parts:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(parts).astype(np.intp, copy=False)


def stable_top_k(
    dists: np.ndarray, k: int, *, value_bound: int | None = None
) -> np.ndarray:
    """Indices of the ``k`` smallest distances, stably tie-broken by index.

    Equivalent to ``np.argsort(dists, kind="stable")[:k]`` but avoids the
    full O(n log n) sort when ``k`` ≪ ``n``.

    Two implementations sit behind this contract, chosen by dtype and size:

    * **Counting select** (:func:`counting_top_k`) for integer distances at
      ``n >= COUNTING_MIN_N``. Hamming distances live in ``[0, 8B]``, so one
      histogram pass locates the cutoff exactly. Allocates ~2 KB of counters
      instead of argpartition's ``8n``-byte permutation.
    * **argpartition + widening** otherwise — including for every float caller
      (``search_asymmetric`` passes ``-scores``), where a histogram is not
      defined.

    Why the widening in the argpartition path: ``argpartition`` is unstable.
    When several distances tie at the kth value, it may keep a higher-indexed
    element inside the top-k partition and exclude a lower-indexed element with
    the same distance. Sorting only inside the partition cannot recover the
    lower-indexed element. Widening to ``dists <= pivot`` brings every tied
    candidate back into scope so the final stable sort matches
    ``argsort(dists, kind="stable")[:k]`` byte-for-byte.

    Both paths return the identical permutation. Which one ran is an
    optimisation detail and must never be observable in the result — that is
    what ``tests/test_query_path.py`` and the query-path gate hold them to.

    Parameters
    ----------
    dists : np.ndarray, shape (n,)
        Per-row distances. Any totally-ordered numeric dtype.
    k : int
        Number of indices to return. Must be positive; the result is
        clamped to ``min(k, n)``.
    value_bound : int, optional, keyword-only
        Largest value ``dists`` can hold, when the caller knows it (``8 * B``
        for a B-byte Hamming code). Only a hint to the counting path, letting
        it skip an ``O(n)`` max reduction. A wrong bound is detected and falls
        back; it cannot produce a wrong answer.

    Returns
    -------
    order : np.ndarray, shape (min(k, n),), dtype intp
        Indices into ``dists`` in ascending distance order, ties broken
        by ascending index.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = dists.shape[0]
    k_eff = min(k, n)
    if k_eff == n:
        return np.argsort(dists, kind="stable")[:k_eff]
    if n >= COUNTING_MIN_N and np.issubdtype(dists.dtype, np.integer):
        try:
            return _counting_top_k(dists, k_eff, value_bound)
        except _NotCountable:
            pass  # negative or out-of-range values — comparison path below
    part = np.argpartition(dists, k_eff - 1)
    pivot = dists[part[k_eff - 1]]
    cand = np.flatnonzero(dists <= pivot)  # ascending order = stable for ties
    return cand[np.argsort(dists[cand], kind="stable")][:k_eff]


# ── blocked multi-query scan ─────────────────────────────────────────────────
#
# The m-query loop in SignBitQuantizer.search reads the whole corpus once per
# query: m full passes over n*B bytes. Past last-level cache that is m times
# the DRAM traffic of the information actually needed, and the corpus rows are
# evicted between queries even though the very next query wants them again.
#
# Inverting the loop nest fixes it at the Python level, with no change to the
# C kernel: hold a block of rows, score ALL m queries against it, move on. The
# block is sized to sit in cache, so it is read from DRAM once and reused m-1
# times. Traffic goes from m*n*B to n*B + (m * per-block-buffer).
#
# The output must stay bit-identical, which constrains the merge. Each block
# contributes its own stable top-k (block-local indices, offset to global), and
# the running list is merged with it by a stable sort on distance. That is
# exact, not approximate:
#
#   * the running list holds only indices < block start, and the new candidates
#     only indices >= block start, so on the concatenation [running, block] a
#     tie group's members are already in ascending global index order;
#   * a stable sort on distance therefore yields (distance, index) order, which
#     is argsort(kind="stable");
#   * truncating to k after each block is safe because an element outside the
#     top-k of everything seen so far can never re-enter — later blocks only
#     add competitors.

#: Target footprint for one corpus block, in bytes. 4 MB is this box's L2 per
#: core; the block is read from DRAM once and then hit m-1 times out of cache.
_BLOCK_TARGET_BYTES = 4 << 20

#: Never block below this many rows — at that point the per-block Python and
#: merge overhead dominates whatever locality is bought.
_MIN_BLOCK_ROWS = 1 << 13


def _resolve_block(block: int | None, n: int, m: int, code_bytes: int) -> int:
    """Rows per corpus block. ``n`` (i.e. no blocking) when it cannot pay."""
    if block is not None:
        if block <= 0:
            raise ValueError(f"block must be positive, got {block}")
        return min(int(block), n)
    if m < 2:
        return n  # one query reads the corpus once regardless
    rows = max(_MIN_BLOCK_ROWS, _BLOCK_TARGET_BYTES // max(1, code_bytes))
    if rows >= n:
        return n  # already fits; blocking would only add bookkeeping
    return int(rows)


def _merge_topk(
    run_idx: np.ndarray,
    run_dist: np.ndarray,
    new_idx: np.ndarray,
    new_dist: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable merge of two (distance, index)-ordered candidate lists.

    ``new_idx`` must hold strictly larger indices than ``run_idx`` — it does,
    because blocks are visited in increasing row order — so a stable sort on
    the concatenated distances reproduces ``argsort(kind="stable")`` exactly.
    """
    if run_idx.size == 0:
        idx, dist = new_idx, new_dist
    else:
        idx = np.concatenate((run_idx, new_idx))
        dist = np.concatenate((run_dist, new_dist))
        order = np.argsort(dist, kind="stable")
        idx, dist = idx[order], dist[order]
    return idx[:k], dist[:k]


def hamming_topk_batch(
    codes: np.ndarray,
    q_codes: np.ndarray,
    k: int,
    *,
    threads: int | str | None = None,
    block: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k Hamming neighbours for a batch of packed queries.

    Bit-identical to looping :func:`hamming_distances` + :func:`stable_top_k`
    per query — that equivalence is the whole contract, and it is what
    ``tests/test_query_path.py`` and the query-path gate check.

    Parameters
    ----------
    codes : np.ndarray, shape (n, B), dtype uint8
        Bit-packed corpus, C-contiguous.
    q_codes : np.ndarray, shape (m, B), dtype uint8
        Bit-packed queries.
    k : int
        Neighbours per query; clamped to ``min(k, n)``.
    threads : int | "auto" | None, keyword-only
        See :func:`hamming_distances`. In the blocked path the parallel unit is
        one (block, query) scan, so a batch keeps every worker fed even when a
        single block is too small to split.
    block : int | None, keyword-only
        Rows per corpus block. ``None`` picks a cache-sized block for ``m >= 2``
        and disables blocking for a single query (where it buys nothing).
        Passing ``block=n`` restores the per-query full-corpus loop exactly.

    Returns
    -------
    indices : np.ndarray, shape (m, min(k, n)), dtype intp
    distances : np.ndarray, shape (m, min(k, n)), dtype int32
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    codes = as_codes(codes)
    q_codes = np.ascontiguousarray(q_codes, dtype=np.uint8)
    if q_codes.ndim != 2 or q_codes.shape[1] != codes.shape[1]:
        raise ValueError(
            f"q_codes shape {q_codes.shape} incompatible with codes shape "
            f"{codes.shape}"
        )
    n, code_bytes = codes.shape
    m = q_codes.shape[0]
    k_eff = min(k, n)
    bound = 8 * code_bytes

    out_idx = np.empty((m, k_eff), dtype=np.intp)
    out_dist = np.empty((m, k_eff), dtype=np.int32)
    if m == 0 or k_eff == 0:
        return out_idx, out_dist

    blk = _resolve_block(block, n, m, code_bytes)
    workers = resolve_threads(threads)

    if blk >= n:
        # Unblocked: one full-corpus scan per query, reusing a single (n,)
        # scratch buffer. This is the pre-existing shape of the loop, kept
        # because for m == 1 there is nothing to amortise.
        scratch = np.empty(n, dtype=np.int32)
        for i in range(m):
            hamming_distances(codes, q_codes[i], out=scratch, threads=threads)
            order = stable_top_k(scratch, k_eff, value_bound=bound)
            out_idx[i] = order
            out_dist[i] = scratch[order]
        return out_idx, out_dist

    # Blocked: corpus read once per block for all m queries.
    buf = np.empty((m, blk), dtype=np.int32)
    run_idx = [np.empty(0, dtype=np.intp) for _ in range(m)]
    run_dist = [np.empty(0, dtype=np.int32) for _ in range(m)]
    pool = _pool(workers) if workers > 1 and m > 1 else None

    for start in range(0, n, blk):
        stop = min(start + blk, n)
        sub = codes[start:stop]
        length = stop - start
        if pool is None:
            for i in range(m):
                hamming_distances(
                    sub, q_codes[i], out=buf[i, :length], threads=1
                )
        else:
            futures = [
                pool.submit(
                    hamming_distances,
                    sub,
                    q_codes[i],
                    out=buf[i, :length],
                    threads=1,
                )
                for i in range(m)
            ]
            for f in futures:
                f.result()
        kk = min(k_eff, length)
        for i in range(m):
            d = buf[i, :length]
            order = stable_top_k(d, kk, value_bound=bound)
            run_idx[i], run_dist[i] = _merge_topk(
                run_idx[i], run_dist[i],
                order.astype(np.intp, copy=False) + start, d[order],
                k_eff,
            )

    for i in range(m):
        out_idx[i] = run_idx[i]
        out_dist[i] = run_dist[i]
    return out_idx, out_dist


# ── asymmetric scoring ───────────────────────────────────────────────────────
#
# Hamming search binarizes BOTH sides. That is symmetric and cheap, but the
# query is a single vector per search -- it occupies no index storage -- so
# throwing away its precision buys nothing. Keeping it in float and scoring it
# against the +/-1 document bits is strictly more information at identical
# storage cost.
#
# Exa's web-scale index (exa.ai/blog/building-web-scale-vector-db) makes this
# choice, and measured on LFM2.5/SciFact it is worth +0.019 nDCG@10 at
# 128 B/vector, widening to +0.084 at 16 B -- the harder you compress, the more
# the query precision is carrying. See bench/results/lfm25_asymmetric.json.
#
# The obvious implementation unpacks the codes to a dense (n, d) +/-1 matrix and
# calls BLAS, but that allocates 32x the index and defeats the point. Instead
# precompute, per byte position, the partial dot product for all 256 possible
# byte values, then score by gather-and-sum. That is Exa's subvector lookup
# table at length 8 rather than length 4: O(n * d/8) table lookups instead of
# O(n * d) multiply-accumulates, with no dense intermediate.

_UNPACK_LUT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).astype(np.float32)  # (256, 8), MSB-first to match np.packbits


def asymmetric_tables(
    queries_rotated: np.ndarray, n_bytes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Byte-value lookup tables for a *batch* of float queries.

    ``asymmetric_scores`` builds one ``(B, 256)`` table per call, and
    ``search_asymmetric`` calls it once per query — so a batch of m queries
    made m separate small GEMMs, each with its own BLAS entry, dispatch and
    output allocation, for an inner dimension of 8. This does the same
    arithmetic as one batched ``(m, B, 8) @ (8, 256)`` GEMM.

    Parameters
    ----------
    queries_rotated : np.ndarray, shape (m, d) or (d,)
        Already-rotated, NOT binarized queries.
    n_bytes : int
        Bytes per code; ``d`` must equal ``8 * n_bytes``.

    Returns
    -------
    tables : np.ndarray, shape (m, n_bytes, 256), float32
        ``tables[i, b, v]`` is the partial dot product of query ``i``'s dims
        in byte ``b`` against the 8 bits of byte value ``v``.
    offsets : np.ndarray, shape (m,), float32
        ``sum(q)`` per query — the affine correction ``asymmetric_scores``
        applies to turn ``q . b`` into ``q . (2b - 1)``.
    """
    q = np.ascontiguousarray(queries_rotated, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    elif q.ndim != 2:
        raise ValueError(
            f"queries_rotated must be 1-D or 2-D, got ndim={q.ndim}"
        )
    if q.shape[1] != n_bytes * 8:
        raise ValueError(
            f"queries have {q.shape[1]} dims but codes carry {n_bytes * 8} bits."
        )
    m = q.shape[0]
    tables = np.ascontiguousarray(
        q.reshape(m, n_bytes, 8) @ _UNPACK_LUT.T  # (m, n_bytes, 256)
    )
    return tables, q.sum(axis=1, dtype=np.float32)


def scores_from_table(
    table: np.ndarray,
    codes: np.ndarray,
    offset: float,
    *,
    chunk: int = 1 << 16,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Gather-and-sum a prebuilt ``(B, 256)`` table over a packed corpus.

    The scan half of :func:`asymmetric_scores`, split out so a batch can build
    its tables once and reuse this. ``offset`` is ``sum(q)``.
    """
    n, n_bytes = codes.shape
    if table.shape != (n_bytes, 256):
        raise ValueError(
            f"table shape {table.shape} does not match codes width {n_bytes} "
            f"(expected ({n_bytes}, 256))"
        )
    cols = np.arange(n_bytes)
    if out is None:
        out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        block = codes[start : start + chunk]
        out[start : start + len(block)] = table[cols, block].sum(axis=1)
    out *= 2.0
    out -= offset
    return out


def asymmetric_scores(
    query_rotated: np.ndarray,
    codes: np.ndarray,
    *,
    chunk: int = 1 << 16,
) -> np.ndarray:
    """Dot product of a float query against sign-bit codes. Higher is better.

    Parameters
    ----------
    query_rotated : np.ndarray, shape (d,)
        Already-rotated query (caller applies ``query @ R``), NOT binarized.
    codes : np.ndarray, shape (n, d // 8), dtype uint8
        Bit-packed corpus from :func:`encode_signs`.
    chunk : int, keyword-only
        Rows scored per batch, bounding the (chunk, d/8) gather buffer.

    Returns
    -------
    scores : np.ndarray, shape (n,), dtype float32
        ``query . s`` where ``s`` is the corpus row decoded to +/-1.

    Notes
    -----
    Codes store ``b = (x > 0)`` as 0/1, while the value they represent is
    ``s = 2b - 1``. So ``q . s == 2 * (q . b) - sum(q)``. The table accumulates
    ``q . b``; the affine correction is applied once at the end. It does not
    change the ranking (``sum(q)`` is constant per query) but it makes the
    returned scores true inner products, so they stay meaningful to callers
    that threshold or fuse them rather than just sorting.
    """
    q = np.ascontiguousarray(query_rotated, dtype=np.float32).ravel()
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got ndim={codes.ndim}")
    if codes.dtype != np.uint8:
        raise ValueError(f"codes must be uint8, got {codes.dtype}")
    n_bytes = codes.shape[1]
    if q.size != n_bytes * 8:
        raise ValueError(
            f"query has {q.size} dims but codes carry {n_bytes * 8} bits."
        )

    # table[b, v] = partial dot product of the 8 dims in byte b against value v
    tables, offsets = asymmetric_tables(q, n_bytes)
    return scores_from_table(tables[0], codes, float(offsets[0]), chunk=chunk)

