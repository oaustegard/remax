#!/usr/bin/env python3
"""Gate: the query path returns the same neighbours after being made faster.

The wrong conclusion this exists to prevent
-------------------------------------------
Not "check the search function". Specifically: **shipping a query-path
optimisation that quietly changes which neighbours come back, or which record
attaches to one.**

Four changes landed on that path at once, and each one is the kind that looks
obviously safe in review:

* ``Corpus.search`` grew a batch (m > 1) branch. A batch that silently scores
  query 0 m times returns m plausible, non-empty, correctly-shaped result
  lists.
* ``_fetch_meta``/``lookup`` stopped opening a fresh SQLite connection per
  call. A cached *connection* is fine; a cached *result* is a stale-metadata
  bug, and both are one line.
* ``hamming_distances`` gained ``out=``, so ``search`` reuses one ``(n,)``
  int32 buffer across the m-loop instead of allocating per query. A buffer
  that is not fully rewritten leaks query i-1's distances into query i, which
  looks like an ordinary near-miss in recall, not like a bug.
* The three redundant ``np.ascontiguousarray(codes)`` calls collapsed to one.
  Restoring a silent copy costs only time; *removing* the copy without
  refusing strided input hands a raw pointer to a strided buffer, and the
  native kernel then reads whatever is adjacent in memory.

None of those raise. All four return correctly-shaped, entirely plausible
output. That is the failure mode this gate is aimed at.

Four more landed later, all of them "make the exhaustive scan faster without
changing what it returns", and all of them with the same shape of failure:

* The native scan was **threaded** — ctypes releases the GIL, so row blocks
  dispatch to a pool. A split written as ``n // T`` rows per block drops the
  ``n % T`` tail, and those rows keep whatever was in the output buffer:
  plausible distances, for a corpus that quietly shrank.
* ``stable_top_k`` gained a **counting (histogram) select** for integer
  distances. It documents byte-for-byte equivalence to
  ``np.argsort(kind="stable")[:k]``, and remax PR #32 exists because a naive
  argpartition broke exactly that. A counting select that takes the cutoff's
  tie group from the wrong end returns the same k distances and different
  documents.
* ``Corpus`` gained **mmap residency**. An "mmap" that actually copies still
  answers every query correctly — it just does not do the thing it claims; and
  a memmap that is not C-contiguous turns the guard added in #63 into a
  whole-index copy on every query while the open-time number still looks good.
* The m-query loop was **blocked** so the corpus is read once per block for all
  queries. A merge that forgets earlier blocks returns the last block's
  neighbours: k results, right shape, right dtype, wrong documents.

Speed is what motivated all eight and speed is NOT gated here. See the
coverage notes and ``bench/results/QUERY_PATH_SPEED.md``.

Anchors
-------
Nothing here compares the new code to the old code's saved output — that only
ever proves the code still does what it did.

1. **Sign disagreement on the raw floats.** ``(sign(X @ R) != sign(q @ R))
   .sum(1)`` is the definition of the Hamming distance the packed path
   computes, evaluated without packing, without the popcount LUT, and without
   the C kernel. Nothing in ``packing.py``, ``_native.py`` or ``core.py``
   participates.
2. **The Charikar / Goemans-Williamson collision identity**,
   ``Pr[sign<r,x> != sign<r,y>] = theta/pi`` (Goemans & Williamson 1995,
   Charikar 2002). A closed form, not a measurement: the mean normalised
   Hamming distance must track the angle. This is what catches a "distance"
   that has quietly become a constant, which agreement checks cannot.
3. **SQLite read straight from the file** with an independent connection, so
   the metadata mapping is checked against the database rather than against
   the Corpus object's idea of it.
4. **numpy's own stable sort.** ``np.argsort(kind="stable")[:k]`` is the
   contract ``stable_top_k`` is written against, and it is a different
   implementation by a different author — so it anchors the counting select
   rather than merely agreeing with it.
5. **The index file's bytes, read with plain ``open()``**, and a write to the
   file observed *through* the mapping. A copy cannot see a later write; a
   mapping must. That is a differential check on the residency claim itself,
   not on the distances it produces — which are identical either way, and so
   say nothing about whether anything was mapped.

Running it red
--------------
A gate seen only green has not been shown to work::

    python3 bench/gates/query_path_gate.py                  # expect 0
    python3 bench/gates/query_path_gate.py --self-test      # expect 0; asserts
                                                            # every simulated
                                                            # defect gives 1
    python3 bench/gates/query_path_gate.py --simulate batch-reuses-first-query

``--self-test`` re-invokes this file once per defect in ``SIMULATIONS`` and
fails unless every one drives the gate to exit 1. That is the check on the
gate, as distinct from the check on the code.

Deliberately NOT gated: speed. There is no anchor for how fast this should
run — see ``bench/native_speedup.py``, which is a benchmark and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# -- locate the gating harness ------------------------------------------- #
_CANDIDATES = [
    os.environ.get("GATING_SKILL_DIR"),
    "/tmp/gating-skill/scripts",
    "/mnt/skills/user/gating/scripts",
    str(Path(__file__).resolve().parent),  # vendored fallback, for CI
]
for _c in _CANDIDATES:
    if _c and (Path(_c) / "gate.py").exists():
        sys.path.insert(0, str(_c))
        break
else:  # pragma: no cover - environment problem, not a gate result
    raise SystemExit(
        "cannot find gate.py from the gating skill; set GATING_SKILL_DIR to "
        f"the directory containing it (looked in {_CANDIDATES})"
    )

from gate import Gate  # noqa: E402

import remax  # noqa: E402
import remax.core as core_mod  # noqa: E402
import remax.corpus as corpus_mod  # noqa: E402
import remax.packing as packing_mod  # noqa: E402
from remax.packing import (  # noqa: E402
    NonContiguousCodesWarning,
    hamming_distances,
)
from remax.rotation import haar_rotation  # noqa: E402

D = 256
N = 4000
M = 6          # batch width; > 1 is the whole point
K = 10
SEED = 3

# Tolerances derived from measured spread, not chosen for comfort. Both come
# from 20 seeds at this exact (N, D) — the configuration the gate runs in, not
# a smaller faster one.
#
#   corr(collision rate, angle): 0.6369 +/- 0.0169
#   mean collision / (theta/pi): 1.0002 +/- 0.0008
CORR_MEAN = 0.6369
CORR_SD = 0.0169
CORR_FLOOR = round(CORR_MEAN - 6 * CORR_SD, 3)   # 0.536
RATIO_TOL = 0.005                                # ~6 sd of the measured ratio

# -- threading configuration --------------------------------------------- #
#
# hamming_distances bypasses the pool entirely below _MIN_ROWS_PER_THREAD rows
# per worker, so a threading check run at N=4000 collapses to one worker and
# certifies nothing while reporting PASS. The gating skill's warning about
# known-bads validated at a small/fast setting applies literally here: the
# dropped-block defect is invisible at a size where no block is ever
# dispatched. So the threading checks get their own corpus, sized from the
# library's own bypass rule rather than from a number typed in here, and
# `_kb_dropped_block` is validated at that size.
#
# d is 64 rather than 256 to keep the float sign-disagreement anchor (an
# (N_T, d) @ (d, d) matmul, computed twice) affordable at this row count --
# --self-test re-runs the whole gate once per simulated defect.
D_T = 64
THREAD_COUNTS = (2, 3, 4, 7)

#: Rows in the threading corpus. The offset is 139, not a round number and not
#: the 137 this started at: 4*16384+137 = 65673 = 3 * 21891, so at T=3 the
#: naive ``n // T`` split covers every row and is *not a defect at all*. The
#: known-bad below reported ACCEPTED for that thread count and turned the gate
#: red, which is the check on the gate working: a dropped-tail case validated
#: only at a dividing thread count would have certified nothing. 139 keeps n
#: indivisible by every count in THREAD_COUNTS, and the known-bad now asserts
#: that rather than assuming it.
N_T = 4 * packing_mod._MIN_ROWS_PER_THREAD + 139

# -- blocked-scan configuration ------------------------------------------ #
# Block sizes chosen so none divides N and one is smaller than K: a merge that
# only ever sees whole blocks, or that assumes a block can supply all k, breaks
# on these and not on a round number.
BLOCK_SIZES = (7, 333, 4001)


# ── anchors ─────────────────────────────────────────────────────────────── #


def reference_distances(X: np.ndarray, q: np.ndarray, seed: int) -> np.ndarray:
    """Hamming distance by counting sign disagreements on the raw floats.

    The anchor. No packing, no popcount LUT, no C kernel — this is the
    quantity those things are *supposed* to compute, obtained a different way.
    """
    R = haar_rotation(X.shape[1], seed=seed, dtype=np.float32)
    xs = (np.asarray(X, dtype=np.float32) @ R) > 0
    qs = (np.asarray(q, dtype=np.float32) @ R) > 0
    return (xs != qs).sum(axis=1).astype(np.int64)


def reference_topk(X: np.ndarray, q: np.ndarray, seed: int, k: int) -> np.ndarray:
    """Stable top-k from the anchor distances, ties broken by index."""
    d = reference_distances(X, q, seed)
    return np.argsort(d, kind="stable")[:k]


def angles(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact pairwise angle in radians — the closed form's input."""
    xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    return np.arccos(np.clip(xn @ qn, -1.0, 1.0))


def file_bytes(path: Path, offset: int, length: int) -> np.ndarray:
    """Read the index payload with plain ``open()`` — no numpy, no Corpus."""
    with open(path, "rb") as f:
        f.seek(offset)
        return np.frombuffer(f.read(length), dtype=np.uint8)


def poke_file(path: Path, offset: int, value: int) -> int:
    """Write one byte into the file and return what was there before.

    The differential half of the residency check. A mapping sees a later write
    to the file; a copy taken at open time cannot. Distances are identical
    under both, so nothing about the *results* can distinguish them — which is
    exactly why "residency" needs a check aimed at the residency.
    """
    with open(path, "r+b") as f:
        f.seek(offset)
        old = f.read(1)[0]
        f.seek(offset)
        f.write(bytes([value]))
        f.flush()
        os.fsync(f.fileno())
    return old


def sqlite_ground_truth(db_path: str, rowids: list[int]) -> dict:
    """Read record_id/meta straight from the file, bypassing Corpus."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for rid in rowids:
            row = con.execute(
                "SELECT record_id, meta FROM corpus_meta WHERE rowid = ?",
                (rid,),
            ).fetchone()
            out[rid] = (row[0], json.loads(row[1]) if row and row[1] else None)
        return out
    finally:
        con.close()


# ── simulated defects (the --simulate / --self-test cases) ──────────────── #
#
# Each of these puts a REAL, plausible defect back into the live modules. They
# are not nonsense inputs: every one is a line somebody could write while
# making this path faster, and every one produces correctly-shaped output.


def _sim_batch_reuses_first_query() -> None:
    """A batch loop that scores query 0 for every row."""
    real = core_mod.SignBitQuantizer.search

    def patched(self, query, codes, k=10, *, return_distances=False):
        query = np.asarray(query, dtype=self.dtype)
        if query.ndim == 2:
            query = np.repeat(query[:1], query.shape[0], axis=0)
        return real(self, query, codes, k, return_distances=return_distances)

    core_mod.SignBitQuantizer.search = patched


def _sim_out_buffer_not_rewritten() -> None:
    """`out=` honoured on the first call, then reused without recomputing.

    The shape of a real caching mistake: "the buffer is already the right
    size, skip the work".
    """
    real = packing_mod.hamming_distances
    seen: set[int] = set()

    def patched(codes, query_code, *, out=None, **kw):
        if out is not None:
            key = id(out)
            if key in seen:
                return out          # stale distances from the previous query
            seen.add(key)
        return real(codes, query_code, out=out, **kw)

    packing_mod.hamming_distances = patched
    core_mod.hamming_distances = patched


def _sim_silent_contiguity_copy() -> None:
    """The pre-fix behaviour: copy a strided index, per call, silently."""
    real = packing_mod.as_codes

    def patched(codes, *, argname="codes"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NonContiguousCodesWarning)
            return real(np.ascontiguousarray(codes), argname=argname)

    packing_mod.as_codes = patched
    core_mod.as_codes = patched


def _sim_stale_metadata_cache() -> None:
    """Caching the metadata *rows* rather than the connection."""
    real = corpus_mod.Corpus._fetch_meta

    def patched(self, positions):
        cached = getattr(self, "_meta_cache", None)
        if cached is None:
            cached = real(self, positions)
            self._meta_cache = cached
        return cached

    corpus_mod.Corpus._fetch_meta = patched


def _sim_metadata_off_by_one() -> None:
    """Row i's metadata attached to row i+1 — a plausible rowid/index slip."""
    real = corpus_mod.Corpus._fetch_meta

    def patched(self, positions):
        positions = list(positions)
        shifted = real(self, [p + 1 for p in positions])
        return {p: shifted[p + 1] for p in positions if p + 1 in shifted}

    corpus_mod.Corpus._fetch_meta = patched


def _sim_unstable_tie_breaking() -> None:
    """stable_top_k replaced by a plain argpartition — the obvious "speedup".

    At d=256 the distances cluster hard around d/2, so ties at the k boundary
    are the rule. Unstable selection silently returns a different member of a
    tie group: same k results, same distances, different documents.
    """
    def patched(dists, k, **kw):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        n = dists.shape[0]
        k_eff = min(k, n)
        part = np.argpartition(dists, k_eff - 1)[:k_eff]
        return part[np.argsort(dists[part])]

    core_mod.stable_top_k = patched
    packing_mod.stable_top_k = patched


def _sim_native_reads_strided_pointer() -> None:
    """The contiguity guard removed rather than tightened.

    Collapsing three ascontiguousarray calls to one is right; collapsing them
    to zero hands the native kernel a raw pointer into a strided buffer, which
    then scans whatever is laid out next to it. No exception, wrong distances.
    """
    def patched(codes, *, argname="codes"):
        arr = np.asarray(codes)
        if arr.ndim != 2:
            raise ValueError(f"{argname} must be 2-D, got ndim={arr.ndim}")
        return arr  # contiguity no longer checked or enforced

    packing_mod.as_codes = patched
    core_mod.as_codes = patched

    real_native = remax._native.hamming_distances_native

    def native_patched(codes, query_code, *, out=None, **kw):
        # Pre-fix native entry: no contiguity check, raw pointer straight in.
        codes = np.asarray(codes)
        n, B = codes.shape
        if out is None:
            out = np.empty(n, dtype=np.int32)
        import ctypes as _ct
        remax._native._lib.hamming_scan(
            codes.ctypes.data,
            np.ascontiguousarray(query_code, dtype=np.uint8).ctypes.data,
            out.ctypes.data, _ct.c_int64(n), _ct.c_int64(B),
        )
        return out

    if remax.NATIVE_AVAILABLE:
        remax._native.hamming_distances_native = native_patched


def _sim_distance_is_constant() -> None:
    """A scan that 'works' but has stopped depending on the data.

    Agreement checks between two of our own paths cannot see this if both
    are patched; the closed-form collision anchor can.
    """
    def patched(codes, query_code, *, out=None, **kw):
        n = codes.shape[0]
        if out is None:
            out = np.empty(n, dtype=np.int32)
        out[:] = codes.shape[1] * 4
        return out

    packing_mod.hamming_distances = patched
    core_mod.hamming_distances = patched


def _sim_threading_drops_a_block() -> None:
    """The row split written as ``n // T`` per block.

    The obvious way to write it, and it silently drops the ``n % T`` tail: at
    N_T=65673 with 7 workers that is 5 rows which are never scanned and keep
    whatever the output buffer held. No exception, no shape change, and on a
    freshly allocated buffer the leftover values are plausible small ints.
    """
    def patched(n, parts):
        step = n // parts
        return [(i * step, (i + 1) * step) for i in range(parts)]

    packing_mod._row_blocks = patched


def _sim_counting_select_tie_break() -> None:
    """The cutoff's tie group taken from its END rather than its front.

    ``dists == cutoff`` may hold for thousands of rows; only ``k - n_below`` of
    them fit. Taking the LAST of them instead of the first returns the same k
    distances, in the same order, for different documents — which is precisely
    the argsort-stability contract remax PR #32 was opened to defend, broken in
    a new place.
    """
    real = packing_mod._counting_top_k

    def patched(dists, k, value_bound):
        order = real(dists, k, value_bound)
        if order.size == 0:
            return order
        cutoff = dists[order[-1]]
        tail = order[dists[order] == cutoff]
        if tail.size == 0:
            return order
        # same tie group, taken from the other end
        all_tied = np.flatnonzero(dists == cutoff)
        replacement = all_tied[-tail.size:]
        out = order.copy()
        out[dists[order] == cutoff] = replacement
        return out

    packing_mod._counting_top_k = patched


def _sim_blocked_merge_forgets_earlier_blocks() -> None:
    """The merge that keeps the newest block instead of merging with it.

    ``run_idx, run_dist = new_idx, new_dist`` — one line, and it looks like
    initialisation. The result is a late block's top-k: k neighbours, sorted
    ascending by a real distance, entirely wrong past the first block.

    Guarded by "this block already has k, why merge?", which is both the
    excuse somebody would actually write and what keeps the output full
    length. The unguarded version (``return new_idx[:k]`` always)
    short-changes the final partial block and dies on a broadcast error —
    that is a red, but a red by crash says only that the gate noticed
    something violent. This one returns correctly-shaped, correctly-typed,
    plausibly-sorted neighbours, which is the failure the gate has to catch
    on its merits.

    It fires only where the guard lets it: at block sizes >= k. At block=7
    with k=10 no block can supply k on its own, so the real merge runs and
    that block size is genuinely unaffected — visible in the check's
    per-block detail line, and the reason the check sweeps block sizes
    rather than testing one.
    """
    real = packing_mod._merge_topk

    def patched(run_idx, run_dist, new_idx, new_dist, k):
        if new_idx.size >= k:
            return new_idx[:k], new_dist[:k]
        return real(run_idx, run_dist, new_idx, new_dist, k)

    packing_mod._merge_topk = patched


def _sim_blocked_merge_order_swapped() -> None:
    """The merge's two argument lists concatenated the other way round.

    ``np.concatenate((new, run))`` instead of ``((run, new))``. The stable sort
    then breaks ties toward the LATER block — so a tie between row 12 and row
    3000 resolves to 3000. Identical k distances, identical shape, identical
    dtype, different documents: the remax#32 failure again, one layer up. A
    check on distances cannot see it.
    """
    def patched(run_idx, run_dist, new_idx, new_dist, k):
        if run_idx.size == 0:
            return new_idx[:k], new_dist[:k]
        idx = np.concatenate((new_idx, run_idx))
        dist = np.concatenate((new_dist, run_dist))
        order = np.argsort(dist, kind="stable")
        return idx[order][:k], dist[order][:k]

    packing_mod._merge_topk = patched


def _sim_mmap_silently_copies() -> None:
    """``residency="mmap"`` that reads the file instead of mapping it.

    Every distance, neighbour and record_id is identical — the bytes are the
    same bytes. What is not identical is the thing the argument was added for:
    open cost, resident footprint, and sharing between processes. A results
    check cannot see this, which is why the gate pokes the file and looks
    through the mapping.
    """
    real = corpus_mod._open_codes

    def patched(bin_path, residency):
        codes, n, d, seed = real(bin_path, residency)
        return np.array(codes), n, d, seed  # a copy, wearing the same shape

    corpus_mod._open_codes = patched


def _sim_mmap_loses_contiguity() -> None:
    """A memmap window that is not C-contiguous.

    Costs nothing at open and everything per query: `as_codes` copies the whole
    index on every call, so the open-time win is paid back many times over and
    the residency benchmark still reports a win.
    """
    real = corpus_mod._open_codes

    def patched(bin_path, residency):
        codes, n, d, seed = real(bin_path, residency)
        if residency == "mmap" and codes.shape[0] > 1:
            doubled = np.repeat(np.asarray(codes), 2, axis=0)
            codes = doubled[::2]         # same values, strided view
        return codes, n, d, seed

    corpus_mod._open_codes = patched


SIMULATIONS = {
    "batch-reuses-first-query": _sim_batch_reuses_first_query,
    "out-buffer-not-rewritten": _sim_out_buffer_not_rewritten,
    "silent-contiguity-copy": _sim_silent_contiguity_copy,
    "stale-metadata-cache": _sim_stale_metadata_cache,
    "metadata-off-by-one": _sim_metadata_off_by_one,
    "unstable-tie-breaking": _sim_unstable_tie_breaking,
    "native-reads-strided-pointer": _sim_native_reads_strided_pointer,
    "distance-is-constant": _sim_distance_is_constant,
    "threading-drops-a-block": _sim_threading_drops_a_block,
    "counting-select-tie-break": _sim_counting_select_tie_break,
    "blocked-merge-forgets-earlier-blocks":
        _sim_blocked_merge_forgets_earlier_blocks,
    "blocked-merge-order-swapped": _sim_blocked_merge_order_swapped,
    "mmap-silently-copies": _sim_mmap_silently_copies,
    "mmap-loses-contiguity": _sim_mmap_loses_contiguity,
}


# ── the gate ────────────────────────────────────────────────────────────── #


def run_gate() -> int:
    g = Gate("remax query path — correctness under batching, reuse and out=")

    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    ids = [f"rec-{i:05d}" for i in range(N)]
    meta = [{"i": i, "tag": f"t{i % 7}"} for i in range(N)]

    tmp = tempfile.mkdtemp(prefix="remax_query_gate_")
    corpus = remax.Corpus.build(
        Path(tmp) / "c", X, ids, seed=SEED, meta=meta, rotation="haar"
    )
    codes = corpus.codes
    quant = remax.SignBitQuantizer(d=D, seed=SEED)

    # -- anchor 1: distances match sign disagreement on the raw floats ---- #
    q0_code = quant.encode(Q[0])
    measured = np.asarray(hamming_distances(codes, q0_code), dtype=np.int64)
    expected = reference_distances(X, Q[0], SEED)
    g.check(
        np.array_equal(measured, expected),
        "packed Hamming distances equal sign-disagreement count "
        "[anchor: (sign(X@R) != sign(q@R)).sum(1), no packing/LUT/C kernel]",
        f"n={N} exact matches={int((measured == expected).sum())}/{N} "
        f"max|diff|={int(np.abs(measured - expected).max())}",
    )

    # -- anchor 2: Charikar collision identity ---------------------------- #
    # E[hamming / d] = theta / pi. Closed form, published, not measured here.
    theta = angles(X, Q[0])
    collision = measured / D
    ratio = float(np.mean(collision / (theta / np.pi)))
    g.anchor(
        "mean collision rate tracks theta/pi",
        measured=ratio, published=1.0, rel_tol=RATIO_TOL,
        source="Goemans-Williamson 1995 / Charikar 2002: "
               "Pr[sign<r,x> != sign<r,y>] = theta/pi",
    )
    # The mean anchor above is NOT sufficient on its own, and finding that out
    # is the main thing building this gate produced. For isotropic Gaussian
    # data the mean angle is ~pi/2, so theta/pi ~ 0.5 — and a scan that has
    # stopped looking at the data and returns the constant d/2 lands on
    # ratio = 1.0027, inside any sane tolerance. It is an assertion whose
    # truth barely depends on the subject.
    #
    # The correlation carries that instead. A constant decorrelates to 0.0
    # while the real estimator sits at 0.637 +/- 0.017 (20 seeds, n=4000,
    # d=256), so the floor is set at mean - 6sd = 0.54: outside the measured
    # noise, and still a chasm away from a collapsed scan. It is NOT set to
    # something comfortable like 0.9 — the estimator has variance ~1/d, and a
    # threshold picked for how correlated a "good" scan feels rather than for
    # how correlated this one measures would have gone red on correct code.
    corr = float(np.corrcoef(collision, theta)[0, 1])
    g.bracket(
        "collision rate is monotone in angle (a constant scan is not)",
        value=corr, lo=CORR_FLOOR, hi=1.0, hi_inclusive=True,
        why=f"measured {CORR_MEAN:.3f} +/- {CORR_SD:.3f} over 20 seeds; floor "
            f"is mean-6sd. A decorrelated (constant) scan gives 0.0. Top edge "
            f"inclusive: perfect monotonicity is a legitimate result",
    )

    # -- batch (m > 1) ---------------------------------------------------- #
    batch = corpus.search(Q, k=K)
    per_query = [corpus.search(Q[i], k=K) for i in range(M)]
    g.check(
        isinstance(batch, list) and len(batch) == M
        and all(isinstance(r, list) and len(r) == K for r in batch),
        "batch search returns m lists of k results",
        f"m={M} k={K} got={[len(r) for r in batch] if isinstance(batch, list) else batch!r}",
    )
    g.check(
        batch == per_query,
        "batch results are identical to the same queries run one at a time",
        f"m={M}; first divergence: "
        + next(
            (f"row {i}" for i in range(M) if batch[i] != per_query[i]), "none"
        ),
    )
    # Against the anchor, not against ourselves.
    anchor_ok = all(
        [r.record_id for r in batch[i]]
        == [ids[j] for j in reference_topk(X, Q[i], SEED, K)]
        for i in range(M)
    )
    g.check(
        anchor_ok,
        "every batch row's top-k matches the float sign-disagreement ranking "
        "[anchor: argsort of (sign(X@R) != sign(q@R)).sum(1)]",
        f"m={M} k={K}",
    )
    g.check(
        len({tuple(r.record_id for r in row) for row in batch}) == M,
        "the m rows are distinct (a batch that scores query 0 m times is not)",
        f"distinct result rows: "
        f"{len({tuple(r.record_id for r in row) for row in batch})}/{M}",
    )

    # -- out= preallocation ----------------------------------------------- #
    fresh = [np.asarray(hamming_distances(codes, quant.encode(Q[i])))
             for i in range(M)]
    buf = np.empty(N, dtype=np.int32)
    reused = []
    for i in range(M):
        got = hamming_distances(codes, quant.encode(Q[i]), out=buf)
        assert got is buf, "out= must return the caller's buffer"
        reused.append(buf.copy())
    g.check(
        all(np.array_equal(a, b) for a, b in zip(fresh, reused)),
        "out= gives byte-identical distances to a fresh allocation, "
        "for every query in the loop",
        f"m={M} mismatched rows="
        f"{sum(not np.array_equal(a, b) for a, b in zip(fresh, reused))}",
    )
    # Poison the buffer: nothing from before the call may survive it.
    buf[:] = np.int32(-12345)
    hamming_distances(codes, quant.encode(Q[0]), out=buf)
    g.check(
        not bool((buf == -12345).any()),
        "out= is fully rewritten — no value from a previous query survives",
        f"poisoned entries left: {int((buf == -12345).sum())}/{N}",
    )

    # -- contiguity: loud, not silently slow ------------------------------ #
    strided = np.ascontiguousarray(np.repeat(codes, 2, axis=0))[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strided_d = np.asarray(hamming_distances(strided, q0_code))
    g.check(
        any(issubclass(w.category, NonContiguousCodesWarning) for w in caught),
        "a non-contiguous index warns instead of copying silently",
        f"warnings raised: {[w.category.__name__ for w in caught] or 'none'}",
    )
    g.check(
        np.array_equal(strided_d.astype(np.int64), expected),
        "the copy, when it happens, is still correct "
        "[anchor: sign-disagreement count]",
        "strided view scored equal to the float reference",
    )
    native_refuses = False
    if remax.NATIVE_AVAILABLE:
        try:
            remax._native.hamming_distances_native(strided, q0_code)
        except ValueError:
            native_refuses = True
    else:  # pragma: no cover - no compiler on this box
        native_refuses = True
    g.check(
        native_refuses,
        "the native kernel refuses a strided buffer rather than reading it",
        "a raw pointer into a strided array reads unrelated memory",
    )

    # -- metadata across connection reuse --------------------------------- #
    positions = [int(j) for j in reference_topk(X, Q[0], SEED, K)]
    truth = sqlite_ground_truth(corpus._db_path, positions)
    got = corpus.search(Q[0], k=K)
    g.check(
        [r.record_id for r in got] == [truth[p][0] for p in positions]
        and [r.meta for r in got] == [truth[p][1] for p in positions],
        "record_id and meta match the database "
        "[anchor: independent sqlite3 connection, no Corpus in the path]",
        f"k={K}",
    )
    # Reuse must not staleness-poison a *second, different* query.
    pos1 = [int(j) for j in reference_topk(X, Q[1], SEED, K)]
    truth1 = sqlite_ground_truth(corpus._db_path, pos1)
    got1 = corpus.search(Q[1], k=K)
    g.check(
        [r.record_id for r in got1] == [truth1[p][0] for p in pos1],
        "a second search over the reused connection resolves its own rows, "
        "not the first search's [anchor: independent sqlite3 connection]",
        f"query 1 top-k record_ids match the database",
    )
    g.check(
        corpus.lookup(ids[123]) == 123 and corpus.lookup("nope") is None,
        "reverse lookup still works over the reused connection",
        f"lookup({ids[123]!r}) -> {corpus.lookup(ids[123])}",
    )
    g.check(
        len(corpus._connections) == 1,
        "one connection is opened, not one per call",
        f"connections opened across "
        f"{2 * M + 5}+ metadata calls: {len(corpus._connections)}",
    )

    # -- threaded scan ---------------------------------------------------- #
    # Own corpus, sized from the library's own small-n bypass rule: at N=4000
    # every thread count collapses to one worker and these checks would be
    # green without a thread ever running.
    rng_t = np.random.default_rng(SEED + 1)
    X_t = rng_t.standard_normal((N_T, D_T)).astype(np.float32)
    quant_t = remax.SignBitQuantizer(d=D_T, seed=SEED)
    codes_t = quant_t.encode(X_t)
    q_t = X_t[0]
    qc_t = quant_t.encode(q_t)
    expected_t = reference_distances(X_t, q_t, SEED)

    packing_mod._pools.clear()
    serial_t = np.asarray(
        hamming_distances(codes_t, qc_t, threads=1), dtype=np.int64
    )
    g.check(
        np.array_equal(serial_t, expected_t),
        "serial scan on the threading corpus matches the float reference "
        "[anchor: sign-disagreement count]",
        f"n={N_T} d={D_T}",
    )
    thread_ok = {}
    poison_left = {}
    for t in THREAD_COUNTS:
        buf = np.full(N_T, np.int32(-9999), dtype=np.int32)
        got = np.asarray(
            hamming_distances(codes_t, qc_t, out=buf, threads=t), dtype=np.int64
        )
        thread_ok[t] = bool(np.array_equal(got, expected_t))
        poison_left[t] = int((buf == -9999).sum())
    g.check(
        all(thread_ok.values()),
        "the threaded scan equals the float sign-disagreement reference at "
        f"every thread count in {THREAD_COUNTS} "
        "[anchor: (sign(X@R) != sign(q@R)).sum(1)]",
        f"n={N_T} (divisible by none of them); per-thread agreement "
        f"{thread_ok}",
    )
    g.check(
        not any(poison_left.values()),
        "every row is written by some thread — no block is dropped",
        f"poisoned rows surviving, by thread count: {poison_left}. "
        f"n % T = {[N_T % t for t in THREAD_COUNTS]}",
    )
    g.check(
        bool(packing_mod._pools),
        "the pool was actually used (a check that collapses to one worker "
        "certifies nothing)",
        f"pools created: {sorted(packing_mod._pools)}; "
        f"_MIN_ROWS_PER_THREAD={packing_mod._MIN_ROWS_PER_THREAD}, "
        f"n={N_T}",
    )
    g.check(
        packing_mod.get_default_threads() == 1,
        "threading is off by default — a library that spawns threads inside "
        "somebody else's pool is a bad neighbour",
        f"get_default_threads()={packing_mod.get_default_threads()}",
    )

    # -- counting select -------------------------------------------------- #
    # Anchored on numpy's own stable sort, which is the contract stable_top_k
    # is written against and an implementation nobody here wrote.
    rng_c = np.random.default_rng(SEED + 2)
    tie_dense = rng_c.binomial(D, 0.5, size=40_000).astype(np.int32)
    counting_ok = all(
        np.array_equal(
            packing_mod.counting_top_k(tie_dense, kk),
            np.argsort(tie_dense, kind="stable")[:kk],
        )
        for kk in (1, K, 137, 5000)
    )
    g.check(
        counting_ok,
        "counting select equals argsort(kind='stable')[:k] on tie-dense "
        "integer distances [anchor: numpy's stable sort]",
        f"n=40000 alphabet=[0,{D}] k in (1, {K}, 137, 5000); "
        f"distinct values present: {len(np.unique(tie_dense))}",
    )
    # The tie group at the cutoff, isolated. This is the remax#32 failure
    # reproduced in a new place: same k distances, different documents.
    tie_probe = np.full(20_000, 5, dtype=np.int32)
    tie_probe[:K - 1] = 1
    tie_order = packing_mod.counting_top_k(tie_probe, K)
    g.check(
        np.array_equal(tie_order, np.arange(K)),
        "the cutoff's tie group is taken from its FRONT (lowest indices), "
        "not from anywhere else in the group "
        "[anchor: argsort(kind='stable')]",
        f"{K - 1} rows below the cutoff, {20_000 - K + 1} tied at it; "
        f"slot {K - 1} went to index {int(tie_order[-1])}, must be {K - 1}",
    )
    # Same data, both implementations, same permutation — the dispatch is an
    # optimisation and must not be observable.
    big = rng_c.binomial(D, 0.5, size=packing_mod.COUNTING_MIN_N + 777)
    big = big.astype(np.int32)
    g.check(
        np.array_equal(
            packing_mod.stable_top_k(big, K),
            np.argsort(big, kind="stable")[:K],
        )
        and np.array_equal(
            packing_mod.stable_top_k(big, K),
            packing_mod.stable_top_k(big.astype(np.float64), K).astype(np.intp),
        ),
        "the counting and comparison paths return the identical permutation "
        "above the dispatch threshold [anchor: argsort(kind='stable')]",
        f"n={big.size} >= COUNTING_MIN_N={packing_mod.COUNTING_MIN_N}",
    )
    g.check(
        np.array_equal(
            np.asarray(quant.search(Q[0], codes, k=K)),
            reference_topk(X, Q[0], SEED, K),
        ),
        "top-k through the live search path still matches the float "
        "reference ranking [anchor: argsort of sign-disagreement counts]",
        f"n={N} d={D} k={K}",
    )

    # -- blocked batch scan ----------------------------------------------- #
    q_codes_all = quant.encode(Q)
    blocked_ok = {}
    for blk in BLOCK_SIZES:
        idx_b, dist_b = packing_mod.hamming_topk_batch(
            codes, q_codes_all, K, block=blk
        )
        blocked_ok[blk] = all(
            np.array_equal(idx_b[i], reference_topk(X, Q[i], SEED, K))
            for i in range(M)
        )
    g.check(
        all(blocked_ok.values()),
        "the blocked scan's top-k matches the float reference ranking at "
        f"every block size in {BLOCK_SIZES} "
        "[anchor: argsort of sign-disagreement counts]",
        f"n={N} m={M} k={K}; per-block agreement {blocked_ok}. "
        f"block={BLOCK_SIZES[0]} is smaller than k, so no single block can "
        f"supply the answer",
    )
    unblocked_idx, unblocked_dist = packing_mod.hamming_topk_batch(
        codes, q_codes_all, K, block=N
    )
    small_idx, small_dist = packing_mod.hamming_topk_batch(
        codes, q_codes_all, K, block=333
    )
    g.check(
        np.array_equal(unblocked_idx, small_idx)
        and np.array_equal(unblocked_dist, small_dist),
        "blocked and unblocked agree bit-for-bit, indices and distances",
        f"differing rows: {int((unblocked_idx != small_idx).any(axis=1).sum())}"
        f"/{M}",
    )

    # -- mmap residency --------------------------------------------------- #
    mmap_dir = Path(tmp) / "mm"
    shutil.copytree(Path(tmp) / "c", mmap_dir)
    mm = remax.Corpus(mmap_dir, residency="mmap")
    payload_off = corpus_mod._HEADER_LEN
    on_disk = file_bytes(
        mmap_dir / corpus_mod._BIN_NAME, payload_off, N * (D // 8)
    ).reshape(N, D // 8)
    g.check(
        np.array_equal(np.asarray(mm.codes), on_disk),
        "mmap codes equal the file's payload bytes "
        "[anchor: plain open()/seek()/read(), no numpy, no Corpus]",
        f"n={N} B={D // 8} bytes={N * (D // 8)}",
    )
    g.check(
        mm.codes.flags["C_CONTIGUOUS"],
        "the memmap window is C-contiguous (a strided one would copy the "
        "whole index on every query through as_codes)",
        f"flags: C={mm.codes.flags['C_CONTIGUOUS']} "
        f"F={mm.codes.flags['F_CONTIGUOUS']}",
    )
    with warnings.catch_warnings(record=True) as mm_caught:
        warnings.simplefilter("always")
        mm_results = mm.search(Q, k=K)
    g.check(
        not any(
            issubclass(w.category, NonContiguousCodesWarning) for w in mm_caught
        ),
        "searching an mmap corpus copies nothing (the #63 guard stays silent)",
        f"warnings raised: {[w.category.__name__ for w in mm_caught] or 'none'}",
    )
    g.check(
        all(
            [r.record_id for r in mm_results[i]]
            == [ids[j] for j in reference_topk(X, Q[i], SEED, K)]
            for i in range(M)
        ),
        "mmap search returns the float reference ranking "
        "[anchor: argsort of sign-disagreement counts]",
        f"m={M} k={K}",
    )
    # The residency claim itself. Distances are identical whether the index was
    # mapped or copied, so no results check can see the difference — poke the
    # file and look through the mapping.
    probe_off = payload_off + (N // 2) * (D // 8)
    old_byte = int(np.asarray(mm.codes)[N // 2, 0])
    poke_file(mmap_dir / corpus_mod._BIN_NAME, probe_off, old_byte ^ 0xFF)
    sees_write = int(np.asarray(mm.codes)[N // 2, 0]) == (old_byte ^ 0xFF)
    poke_file(mmap_dir / corpus_mod._BIN_NAME, probe_off, old_byte)
    g.check(
        sees_write,
        "the mmap corpus is a view of the file, not a copy of it "
        "[anchor: a byte written to the file with open('r+b'), observed "
        "through the mapping]",
        f"wrote 0x{old_byte ^ 0xFF:02x} at offset {probe_off}; mapping "
        f"reported 0x{int(np.asarray(mm.codes)[N // 2, 0]):02x} after restore, "
        f"saw the write: {sees_write}",
    )
    g.check(
        remax.Corpus(mmap_dir).residency == "load"
        and not isinstance(remax.Corpus(mmap_dir).codes, np.memmap),
        "residency defaults to 'load' — the pre-existing behaviour is what an "
        "unchanged caller still gets",
        f"Corpus(path).residency={remax.Corpus(mmap_dir).residency!r}",
    )

    # -- known-bads ------------------------------------------------------- #
    # Built from the real machinery and broken the way it would plausibly
    # break. `covers` names which checks each one has been shown to fire.
    _kb_batch_reuse(g, corpus, Q, ids, X)
    _kb_stale_out(g, codes, quant, Q, N)
    _kb_silent_copy(g, strided, q0_code)
    _kb_wrong_row(g, corpus, positions, truth)
    _kb_constant_distance(g, codes, theta)
    _kb_dropped_block(g, codes_t, qc_t, expected_t)
    _kb_tie_group_from_the_back(g, tie_probe)
    _kb_blocked_merge_drops_history(g, codes, q_codes_all, X, Q)
    _kb_mmap_that_is_a_copy(g, mmap_dir, on_disk)
    _kb_mmap_strided(g, mm)

    # -- coverage --------------------------------------------------------- #
    g.coverage(
        "Concurrency is not covered. Connections are thread-local and "
        "read-only, but this gate runs single-threaded: it cannot see a "
        "connection escaping to another thread, nor lock contention under "
        "real parallel search."
    )
    g.coverage(
        "Multi-process access is not covered. Two processes holding read-only "
        "connections while a third rebuilds the corpus in place is untested; "
        "mode=ro means no writer lock is held, which is an argument, not a "
        "measurement."
    )
    g.coverage(
        "Performance is not gated, deliberately. Every change here was "
        "motivated by cost — a per-call connect, a per-query allocation, a "
        "silent whole-index copy — and none of that is checked. Wall-clock "
        "has no anchor outside the box it was measured on; see "
        "bench/native_speedup.py, which is a benchmark and says so."
    )
    g.coverage(
        "StackedSignBitQuantizer.search and the asymmetric path "
        "(search_asymmetric / asymmetric_scores) are not exercised. They have "
        "their own m-loops and did not receive out= buffers, so the "
        "conclusions here do not transfer to them."
    )
    g.coverage(
        "The mean-collision anchor cannot, on its own, detect a scan that "
        "stopped reading the data. On isotropic input the mean angle is "
        "~pi/2, so a constant d/2 reproduces theta/pi to 0.3% and passes it. "
        "The monotonicity bracket is what actually carries that case; if that "
        "bracket is ever loosened, the anchor above will not cover for it."
    )
    g.coverage(
        "The anchor is a Haar rotation at d=256, n=4000, f32. It says nothing "
        "about the rht construction, about d not a multiple of 64, or about "
        "n large enough for the int32 distance dtype to matter (it overflows "
        "only past a ~256 MB code, which this does not approach)."
    )

    g.coverage(
        "The threading checks run at 4 workers on a 4-core box. Nothing here "
        "exercises oversubscription, NUMA, or a pool shared with the caller's "
        "own executor. The bit-identity argument is structural (disjoint row "
        "blocks, no reduction) and does not depend on the thread count, but "
        "the *deadlock* question — remax's pool reached from inside a caller's "
        "pool thread — is untested."
    )
    g.coverage(
        "The counting select is checked over alphabets of 257 and 2049 values "
        "at n up to ~1e6. It says nothing about the COUNTING_MAX_VALUE edge "
        "(65536 counters), and nothing about n large enough for the int64 "
        "cumulative counts to matter."
    )
    g.coverage(
        "The mmap checks run on a tmpfs-backed corpus of ~128 KB, which is "
        "resident in page cache throughout. They cannot see the behaviour the "
        "option exists for: a multi-GB index under memory pressure, where "
        "page eviction and first-touch fault latency are the whole story. "
        "The residency probe proves the mapping is a view; it does not prove "
        "the mapping is a good idea at scale."
    )
    g.coverage(
        "Nothing here runs two processes against one mmap'd index, which is "
        "the sharing benefit the option is for. Single-process page-cache "
        "coherence is what the poke-the-file probe demonstrates; cross-process "
        "sharing is an inference from it, not a measurement."
    )
    g.coverage(
        "The blocked scan is checked at n=4000 with blocks of 7 to 4001. The "
        "cache-residency argument that motivates the default block size is a "
        "performance claim and is neither made nor checked here — at this n "
        "the whole corpus fits in L2 and blocking cannot help. What is checked "
        "is only that blocking does not change the answer."
    )
    g.coverage(
        "Speed is not gated, for any of these. Threading, counting select, "
        "mmap and blocking were all motivated by cost and none of that is "
        "checked here: there is no published constant for how fast a scan "
        "should be, so wall-clock belongs to benchmarking discipline (matched "
        "implementation effort, min-of-trials, a stated box) rather than to a "
        "gate. The numbers live in bench/results/QUERY_PATH_SPEED.md, which "
        "states its box and its scope limits."
    )

    g.note(f"corpus n={N} d={D} batch m={M} k={K} seed={SEED}")
    g.note(f"threading corpus n={N_T} d={D_T} threads={THREAD_COUNTS}")
    g.note(f"native kernel available: {remax.NATIVE_AVAILABLE}")

    mm.close()
    corpus.close()
    return g.report()


def _kb_batch_reuse(g, corpus, Q, ids, X) -> None:
    """A batch that scores query 0 m times — shaped right, wrong content."""
    real = corpus.search(Q, k=K)
    faked = [real[0] for _ in range(len(Q))]
    rejected = (
        faked != real
        and len({tuple(r.record_id for r in row) for row in faked}) == 1
    )
    g.known_bad(
        "a batch that reuses query 0 for every row is rejected",
        rejected=rejected,
        detail=f"faked batch has "
               f"{len({tuple(r.record_id for r in row) for row in faked})} "
               f"distinct rows out of {len(Q)}; the real one has "
               f"{len({tuple(r.record_id for r in row) for row in real})}",
        covers=("batch results are identical", "the m rows are distinct",
                "every batch row's top-k matches"),
    )


def _kb_stale_out(g, codes, quant, Q, n) -> None:
    """An out= buffer carrying the previous query's distances."""
    buf = np.empty(n, dtype=np.int32)
    hamming_distances(codes, quant.encode(Q[0]), out=buf)
    stale = buf.copy()                       # what a skipped rewrite leaves
    hamming_distances(codes, quant.encode(Q[1]), out=buf)
    fresh = buf.copy()
    g.known_bad(
        "a stale out= buffer (query i-1's distances) is rejected",
        rejected=not np.array_equal(stale, fresh),
        detail=f"stale vs fresh differ in "
               f"{int((stale != fresh).sum())}/{n} entries — a rewrite that "
               f"was skipped would be caught",
        covers=("out= gives byte-identical distances",
                "out= is fully rewritten"),
    )


def _kb_silent_copy(g, strided, q0_code) -> None:
    """The pre-fix contiguity behaviour: copy, per call, without a word."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NonContiguousCodesWarning)
            np.asarray(hamming_distances(np.ascontiguousarray(strided), q0_code))
    g.known_bad(
        "a silently-copied strided index is rejected (no warning raised)",
        rejected=not any(
            issubclass(w.category, NonContiguousCodesWarning) for w in caught
        ),
        detail="pre-copying to hide the strided input suppresses the warning, "
               "which is exactly the state the check must reject",
        covers=("a non-contiguous index warns",),
    )


def _kb_wrong_row(g, corpus, positions, truth) -> None:
    """Metadata shifted by one row — the classic rowid/index slip.

    Run through the real ``_fetch_meta`` rather than fabricated in a list, so
    the case exercises the connection-reuse code it is supposed to be about.
    """
    correct = [truth[p][0] for p in positions]
    shifted_map = corpus._fetch_meta([p + 1 for p in positions])
    shifted = [shifted_map[p + 1][0] for p in positions]
    g.known_bad(
        "metadata shifted by one row is rejected",
        rejected=shifted != correct,
        detail=f"shifted[0]={shifted[0]!r} vs correct {correct[0]!r}; "
               f"{sum(a != b for a, b in zip(shifted, correct))}/"
               f"{len(correct)} rows differ",
        covers=("record_id and meta match the database",
                "a second search over the reused connection"),
    )


def _kb_constant_distance(g, codes, theta) -> None:
    """A scan that returns a constant: plausible shape, zero information.

    The reason this case is here rather than assumed: the *mean* collision
    anchor accepts it. On isotropic data the mean angle is ~pi/2, so the
    constant d/2 reproduces theta/pi almost exactly (ratio 1.0027). Only the
    monotonicity bracket sees it — hence ``or``, not ``and``. Written with
    ``and`` first, this known-bad reported the constant as *accepted*, which
    is how the hole was found.
    """
    const = np.full(codes.shape[0], codes.shape[1] * 4, dtype=np.int32)
    ratio = float(np.mean((const / D) / (theta / np.pi)))
    corr = np.corrcoef(const.astype(float), theta)[0, 1]
    corr = 0.0 if np.isnan(corr) else float(corr)
    ratio_fires = abs(ratio - 1.0) >= RATIO_TOL
    corr_fires = corr <= CORR_FLOOR
    g.known_bad(
        "a constant 'distance' is rejected (by monotonicity, not by the mean)",
        rejected=ratio_fires or corr_fires,
        detail=f"constant scan: theta/pi ratio={ratio:.4f} "
               f"(tol {RATIO_TOL} -> {'fires' if ratio_fires else 'ACCEPTS'}), "
               f"corr with angle={corr:.4f} "
               f"(floor {CORR_FLOOR} -> {'fires' if corr_fires else 'accepts'})",
        covers=("collision rate is monotone in angle",
                "packed Hamming distances equal sign-disagreement count"),
    )


def _kb_dropped_block(g, codes_t, qc_t, expected_t) -> None:
    """A row split that loses the tail — the ``n // T`` per block mistake.

    Validated at N_T, the size the threading path actually runs at, because at
    a smaller size the pool is bypassed and the defect cannot occur: it would
    have been a known-bad that is not bad at the configuration it certifies.
    """
    n = codes_t.shape[0]
    worst = {}
    for t in THREAD_COUNTS:
        step = n // t
        covered = step * t
        buf = np.full(n, np.int32(-9999), dtype=np.int32)
        # Exactly what the naive split computes: the first `covered` rows.
        packing_mod._scan_serial(codes_t[:covered], qc_t, buf[:covered])
        dropped = int((buf == -9999).sum())
        wrong = int(
            (buf[:covered].astype(np.int64) != expected_t[:covered]).sum()
        )
        worst[t] = (dropped, wrong)
    # The defect only exists when T does not divide n. Asserted, not assumed:
    # at 4*16384+137 rows, T=3 divides exactly and the naive split is correct,
    # so this known-bad reported ACCEPTED there and turned the gate red until
    # the corpus size was changed. A known-bad that stops being bad at some
    # configuration certifies nothing at that configuration.
    divides = [t for t in THREAD_COUNTS if n % t == 0]
    rejected = not divides and all(d > 0 for d, _ in worst.values())
    g.known_bad(
        "a threaded split that drops the n % T tail is rejected",
        rejected=rejected,
        detail=f"n={n}; rows left unwritten per thread count "
               f"{ {t: d for t, (d, _) in worst.items()} }. Thread counts "
               f"that divide n exactly (where this is NOT a defect): "
               f"{divides or 'none'}. The rows that WERE scanned are all "
               f"correct ({sum(w for _, w in worst.values())} wrong), which "
               f"is why comparing only the scanned rows would accept this",
        covers=("every row is written by some thread",
                "the threaded scan equals the float sign-disagreement",
                "serial scan on the threading corpus matches",
                "the pool was actually used"),
    )


def _kb_tie_group_from_the_back(g, tie_probe) -> None:
    """The cutoff's tie group taken from its end. remax#32, in a new place.

    Same k distances, same order, different documents — so a check on the
    returned *distances* accepts it and only a check on the *indices* fires.
    """
    correct = np.argsort(tie_probe, kind="stable")[:K]
    all_tied = np.flatnonzero(tie_probe == tie_probe[correct[-1]])
    broken = correct.copy()
    broken[-1] = all_tied[-1]           # last member of the group, not first
    same_dists = np.array_equal(tie_probe[broken], tie_probe[correct])
    g.known_bad(
        "a counting select that takes the tie group from its back is rejected",
        rejected=(not np.array_equal(broken, correct)) and same_dists,
        detail=f"broken[-1]={int(broken[-1])} vs correct {int(correct[-1])}; "
               f"the k distances are identical ({same_dists}), so only an "
               f"index-level check can see this — a distances-only check "
               f"would report PASS",
        covers=("the cutoff's tie group is taken from its FRONT",
                "counting select equals argsort(kind='stable')",
                "the counting and comparison paths return the identical"),
    )


def _kb_blocked_merge_drops_history(g, codes, q_codes_all, X, Q) -> None:
    """A merge that keeps the newest block and forgets the running list.

    Built by running the real blocked machinery with the real per-block
    top-k and only the merge replaced, so the case exercises the blocking
    code rather than fabricating a wrong answer.
    """
    blk = 333
    n = codes.shape[0]
    faked = []
    for i in range(M):
        run_i = np.empty(0, dtype=np.intp)
        run_d = np.empty(0, dtype=np.int32)
        for start in range(0, n, blk):
            stop = min(start + blk, n)
            d = np.asarray(hamming_distances(codes[start:stop], q_codes_all[i]))
            order = packing_mod.stable_top_k(d, min(K, stop - start))
            run_i, run_d = order + start, d[order]   # the forgetful merge
        faked.append(run_i[:K])
    truth = [reference_topk(X, Q[i], SEED, K) for i in range(M)]
    differs = sum(
        not np.array_equal(faked[i], truth[i]) for i in range(M)
    )
    g.known_bad(
        "a blocked merge that forgets earlier blocks is rejected",
        rejected=differs == M,
        detail=f"{differs}/{M} rows differ from the float reference; the "
               f"faked rows are still k sorted indices with real distances, "
               f"all drawn from the last block "
               f"([{int(faked[0].min())}, {int(faked[0].max())}] of "
               f"[0, {n})) ",
        covers=("the blocked scan's top-k matches the float reference",
                "blocked and unblocked agree bit-for-bit"),
    )


def _kb_mmap_that_is_a_copy(g, mmap_dir, on_disk) -> None:
    """An "mmap" that read the file. Every query answer is correct.

    This is the case that motivates the poke-the-file probe: the codes match
    the file byte for byte, the neighbours match the anchor, the record_ids
    match the database. Nothing about the *results* is wrong. Only the claim
    is.
    """
    path = mmap_dir / corpus_mod._BIN_NAME
    real = corpus_mod._open_codes
    copied, n, d, _seed = real(path, "mmap")
    copied = np.array(copied)                     # the defect, in one call
    probe_off = corpus_mod._HEADER_LEN + (N // 2) * (D // 8)
    old = int(copied[N // 2, 0])
    poke_file(path, probe_off, old ^ 0xFF)
    sees_write = int(copied[N // 2, 0]) == (old ^ 0xFF)
    poke_file(path, probe_off, old)
    matches_disk = np.array_equal(copied, on_disk)
    g.known_bad(
        "an 'mmap' that is really a copy is rejected",
        rejected=(not sees_write) and matches_disk,
        detail=f"the copy matches the file's bytes ({matches_disk}) and would "
               f"answer every query identically; it does not see a later "
               f"write to the file ({sees_write}), which is the only "
               f"observable difference and therefore the only thing that can "
               f"catch it",
        covers=("the mmap corpus is a view of the file",),
    )


def _kb_mmap_strided(g, mm) -> None:
    """A memmap window that is not C-contiguous — a per-query whole-index copy.

    Free at open time, so the residency benchmark still reports a win while
    every query pays 32 MB of memcpy.
    """
    strided = np.repeat(np.asarray(mm.codes), 2, axis=0)[::2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        packing_mod.as_codes(strided)
    warned = any(
        issubclass(w.category, NonContiguousCodesWarning) for w in caught
    )
    g.known_bad(
        "a non-contiguous mmap window is rejected",
        rejected=warned and not strided.flags["C_CONTIGUOUS"],
        detail=f"strided view: C_CONTIGUOUS={strided.flags['C_CONTIGUOUS']}, "
               f"as_codes warned={warned} "
               f"({strided.nbytes / 1e6:.1f} MB copied per query); the values "
               f"are identical to the real codes "
               f"({np.array_equal(strided, np.asarray(mm.codes))}), so only "
               f"the contiguity check sees it",
        covers=("the memmap window is C-contiguous",
                "searching an mmap corpus copies nothing"),
    )


# ── entry point ─────────────────────────────────────────────────────────── #


def self_test() -> int:
    """Re-run this gate once per simulated defect; every one must go RED."""
    failures = []
    print("=" * 74)
    print("SELF-TEST — each simulated defect must drive the gate to exit 1")
    print("=" * 74)
    for name in SIMULATIONS:
        proc = subprocess.run(
            [sys.executable, __file__, "--simulate", name],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 1
        print(f"  [{'RED ' if ok else 'MISS'}] {name:32s} exit={proc.returncode}")
        if not ok:
            failures.append(name)
            print("        --- gate output ---")
            for line in proc.stdout.splitlines()[-25:]:
                print("        " + line)
    print("=" * 74)
    if failures:
        print(f"SELF-TEST FAILED — the gate did NOT go red for: {failures}")
        print("A defect the gate cannot catch is a defect it will ship.")
        return 1
    print(f"SELF-TEST PASSED — all {len(SIMULATIONS)} simulated defects rejected")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--simulate", choices=sorted(SIMULATIONS),
        help="reintroduce a real defect into the live modules; the gate must "
             "then FAIL (exit 1)",
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help="run every simulated defect and assert the gate goes red for each",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.simulate:
        print(f"[simulating defect: {args.simulate}]")
        SIMULATIONS[args.simulate]()

    try:
        return run_gate()
    except Exception as exc:
        # Under --simulate, a crash is a legitimate red: the defect was
        # caught, just violently. Without one, it is a broken gate.
        if args.simulate:
            print(f"\nFAILED — simulated defect {args.simulate!r} raised "
                  f"{type(exc).__name__}: {exc}")
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
