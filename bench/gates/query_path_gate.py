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

    def patched(codes, query_code, *, out=None):
        if out is not None:
            key = id(out)
            if key in seen:
                return out          # stale distances from the previous query
            seen.add(key)
        return real(codes, query_code, out=out)

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
    def patched(dists, k):
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

    def native_patched(codes, query_code, *, out=None):
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
    def patched(codes, query_code, *, out=None):
        n = codes.shape[0]
        if out is None:
            out = np.empty(n, dtype=np.int32)
        out[:] = codes.shape[1] * 4
        return out

    packing_mod.hamming_distances = patched
    core_mod.hamming_distances = patched


SIMULATIONS = {
    "batch-reuses-first-query": _sim_batch_reuses_first_query,
    "out-buffer-not-rewritten": _sim_out_buffer_not_rewritten,
    "silent-contiguity-copy": _sim_silent_contiguity_copy,
    "stale-metadata-cache": _sim_stale_metadata_cache,
    "metadata-off-by-one": _sim_metadata_off_by_one,
    "unstable-tie-breaking": _sim_unstable_tie_breaking,
    "native-reads-strided-pointer": _sim_native_reads_strided_pointer,
    "distance-is-constant": _sim_distance_is_constant,
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

    # -- known-bads ------------------------------------------------------- #
    # Built from the real machinery and broken the way it would plausibly
    # break. `covers` names which checks each one has been shown to fire.
    _kb_batch_reuse(g, corpus, Q, ids, X)
    _kb_stale_out(g, codes, quant, Q, N)
    _kb_silent_copy(g, strided, q0_code)
    _kb_wrong_row(g, corpus, positions, truth)
    _kb_constant_distance(g, codes, theta)

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

    g.note(f"corpus n={N} d={D} batch m={M} k={K} seed={SEED}")
    g.note(f"native kernel available: {remax.NATIVE_AVAILABLE}")

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
