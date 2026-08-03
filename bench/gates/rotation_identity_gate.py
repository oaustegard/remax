#!/usr/bin/env python3
"""Gate: a stored corpus decodes the same way after the library default flips.

The hazard this guards
----------------------
``SignBitQuantizer`` can rotate with either the Haar QR construction or a
randomized Hadamard transform. The two give *different codes* from the same
``(d, seed)``, and until ``rotation.json`` existed nothing on disk recorded
which one wrote a corpus. So the reader had to fall back on the library's
current default — and the day somebody flips that default, every already-stored
corpus starts decoding queries into a rotation frame its codes were never
written in. Nothing raises. Search keeps returning ten neighbours; they are
just the wrong ten.

``bench/results/ROTATION_LSH.md`` names this as the reason ``"haar"`` could not
be changed. This gate is what makes the default *safe to flip*: it asserts that
flipping it is a no-op for decode.

What is checked
---------------
A corpus is built with the real library, then its ``rotation.json`` is deleted
to manufacture a pre-sidecar (legacy) index — the exact artifact whose
reinterpretation is the hazard. The library default is then flipped to ``rht``
by rebinding ``SignBitQuantizer.__init__.__kwdefaults__``, and the legacy corpus
is reopened. Codes, query encoding and search results must be **byte-identical**
across that flip, and identical to a Haar reference computed straight from
``remax.rotation.haar_rotation`` + ``remax.packing.encode_signs`` without going
through ``corpus.py`` at all.

Running it red
--------------
``--simulate-prefix`` patches ``remax.corpus._read_rotation`` to return the live
module default instead of reading the sidecar, which is precisely the pre-fix
reader. The gate must FAIL under that flag; a run that has only ever been seen
green has not been shown to work.

    python3 bench/gates/rotation_identity_gate.py                    # expect 0
    python3 bench/gates/rotation_identity_gate.py --simulate-prefix  # expect 1

The harness lives in the ``gating`` skill; point ``GATING_SKILL_DIR`` at its
``scripts/`` directory if it is not staged at ``/tmp/gating-skill/scripts``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# -- locate the gating harness ------------------------------------------- #
_CANDIDATES = [
    os.environ.get("GATING_SKILL_DIR"),
    "/tmp/gating-skill/scripts",
    "/mnt/skills/user/gating/scripts",
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

import remax.corpus as corpus_mod  # noqa: E402
from remax.core import SignBitQuantizer  # noqa: E402
from remax.corpus import Corpus  # noqa: E402
from remax.packing import encode_signs  # noqa: E402
from remax.rotation import haar_rotation, rht_rotation  # noqa: E402

D = 64
N = 200
SEED = 7
K = 10


def _digest(*arrays: np.ndarray) -> str:
    """Byte digest of the exact bytes, not of a float summary."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _reference_codes(X: np.ndarray, kind: str) -> np.ndarray:
    """Codes computed from the rotation primitives, bypassing corpus.py.

    This is the anchor: nothing in ``corpus.py`` participates in producing it,
    so agreement is not the index agreeing with itself.
    """
    fn = haar_rotation if kind == "haar" else rht_rotation
    R = fn(X.shape[1], seed=SEED, dtype=np.float32)
    return encode_signs(np.asarray(X, dtype=np.float32) @ R)


def _flip_library_default(kind: str) -> str:
    """Rebind the quantizer's ``rotation`` default; return the previous value.

    This is the flip the whole gate is about — the one-line change a future
    contributor makes after reading that RHT is 1.5-1.8x faster and measured
    equivalent for retrieval.
    """
    kw = SignBitQuantizer.__init__.__kwdefaults__
    previous = kw["rotation"]
    kw["rotation"] = kind
    return previous


class _Observation(tuple):
    """What a reopened corpus decodes to: rotation, digests, re-encoded codes."""

    __slots__ = ()
    rotation = property(lambda self: self[0])
    query_enc = property(lambda self: self[1])
    results = property(lambda self: self[2])
    recodes = property(lambda self: self[3])


def _observe(path: Path, query: np.ndarray, X: np.ndarray) -> _Observation:
    """Reopen a stored corpus and record what it decodes to."""
    c = Corpus(path)
    enc = c._quantizer.encode(query[None, :])
    results = c.search(query, k=K)
    ids = np.array([r.record_id for r in results], dtype="U16")
    dists = np.array([r.distance for r in results], dtype=np.int64)
    # Re-encoding the original vectors with the reader's own quantizer is the
    # sharpest statement of "the reader is in the frame the codes were written
    # in": it must reproduce the stored codes bit for bit.
    recodes = np.asarray(c._quantizer.encode(X))
    return _Observation((c.rotation, _digest(enc), _digest(ids, dists), recodes))


def _bit_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.unpackbits(a, axis=1) == np.unpackbits(b, axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--simulate-prefix",
        action="store_true",
        help="patch the reader to trust the module default instead of the "
        "sidecar (pre-fix behaviour); the gate MUST go red.",
    )
    args = ap.parse_args()

    g = Gate("rotation identity across a library default flip")

    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, D)).astype(np.float32)
    query = X[3].copy()
    ids = [f"doc-{i:04d}" for i in range(N)]

    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "legacy"
        rht_built = Path(tmp) / "rht"

        # A pre-sidecar index: built by the real library, then stripped of the
        # sidecar. This is the artifact already sitting on users' disks.
        Corpus.build(legacy, X, ids, d=D, seed=SEED)
        (legacy / corpus_mod._ROTATION_NAME).unlink()

        # An index that explicitly asked for the non-default construction.
        Corpus.build(rht_built, X, ids, d=D, seed=SEED, rotation="rht")

        if args.simulate_prefix:
            corpus_mod._read_rotation = lambda p: (
                SignBitQuantizer.__init__.__kwdefaults__["rotation"]
            )
            g.note("SIMULATING PRE-FIX READER: _read_rotation returns the "
                   "live module default, ignoring the sidecar")

        # ---- anchor: the stored codes, computed without corpus.py -------- #
        ref_haar = _reference_codes(X, "haar")
        ref_rht = _reference_codes(X, "rht")
        # Instrument check first: the bytes on disk really are the Haar
        # reference, so a later disagreement is the reader's fault and not the
        # fixture's. This one is NOT sensitive to the bug -- codes are read
        # straight off disk -- so the known-bad below does not claim it.
        stored = np.asarray(Corpus(legacy).codes)
        g.anchor(
            "fixture: stored codes are the Haar reference",
            measured=_bit_agreement(stored, ref_haar),
            published=1.0,
            rel_tol=1e-12,
            source="sign(X @ haar_rotation(d, seed)) packed by "
            "remax.packing.encode_signs — no corpus.py in the path",
        )

        # The two constructions must genuinely disagree, or byte-identity
        # across the flip would be trivially satisfied and prove nothing.
        disagree = 1.0 - _bit_agreement(ref_haar, ref_rht)
        g.bracket(
            "haar and rht codes disagree like independent signs",
            value=disagree,
            lo=0.35,
            hi=0.65,
            why="two independent random hyperplane sets agree on ~half the "
            "bits; this is the size of the damage a misread does, not a "
            "rounding difference",
        )

        # ---- the property: decode is invariant to the library default ---- #
        before = _observe(legacy, query, X)
        previous = _flip_library_default("rht")
        try:
            after = _observe(legacy, query, X)
            g.check(
                after.rotation == "haar",
                "legacy corpus resolves to haar with the default flipped",
                f"resolved={after.rotation!r}, library default={'rht'!r}",
            )
            g.check(
                before.query_enc == after.query_enc,
                "query encoding is byte-identical across the flip",
                f"{before.query_enc[:16]} vs {after.query_enc[:16]}",
            )
            g.check(
                before.results == after.results,
                "search results are byte-identical across the flip",
                f"{before.results[:16]} vs {after.results[:16]}",
            )
            # The codes themselves, against the external reference, read by a
            # reader whose library default now says rht.
            g.anchor(
                "post-flip reader re-encodes to the Haar reference codes",
                measured=_bit_agreement(after.recodes, ref_haar),
                published=1.0,
                rel_tol=1e-12,
                source="sign(X @ haar_rotation(d, seed)) packed by "
                "remax.packing.encode_signs — no corpus.py in the path",
            )

            # The sidecar has to be *read*, not assumed: an rht corpus must
            # still come back as rht. Without this, hardcoding "haar" in the
            # reader would pass everything above.
            rht_obs = _observe(rht_built, query, X)
            rht_rot, rht_enc = rht_obs.rotation, rht_obs.query_enc
            g.check(
                rht_rot == "rht",
                "an rht-built corpus reports rht (sidecar is read, not assumed)",
                f"resolved={rht_rot!r}",
            )
            ref_q_rht = _digest(
                encode_signs(
                    query[None, :].astype(np.float32)
                    @ rht_rotation(D, seed=SEED, dtype=np.float32)
                )
            )
            g.check(
                rht_enc == ref_q_rht,
                "rht corpus encodes queries in the rht frame [anchor: "
                "rht_rotation primitive]",
                f"{rht_enc[:16]} vs {ref_q_rht[:16]}",
            )

            # ---- known-bad: the pre-fix reader ------------------------- #
            # Same machinery, broken the one way it was actually broken:
            # resolve the rotation from the live module default rather than
            # from disk. Every check above must reject this.
            real_read = corpus_mod._read_rotation
            corpus_mod._read_rotation = lambda p: (
                SignBitQuantizer.__init__.__kwdefaults__["rotation"]
            )
            try:
                bad = _observe(legacy, query, X)
            finally:
                corpus_mod._read_rotation = real_read
            bad_agree = _bit_agreement(bad.recodes, ref_haar)
            g.known_bad(
                "pre-fix reader (trusts module default) is rejected",
                rejected=(
                    bad.rotation != "haar"
                    and bad.query_enc != before.query_enc
                    and bad.results != before.results
                    and abs(bad_agree - 1.0) >= 1e-12
                ),
                detail=(
                    f"resolved={bad.rotation!r} (want haar), "
                    f"query-enc "
                    f"{'differs' if bad.query_enc != before.query_enc else 'MATCHES'}, "
                    f"results "
                    f"{'differ' if bad.results != before.results else 'MATCH'}, "
                    f"re-encode agreement with the Haar reference "
                    f"{bad_agree:.4f} (want 1.0)"
                ),
                covers=(
                    "legacy corpus resolves to haar",
                    "query encoding is byte-identical",
                    "search results are byte-identical",
                    "post-flip reader re-encodes to the Haar reference codes",
                ),
            )
            g.note(f"before-flip result digest {before.results[:16]}")
        finally:
            _flip_library_default(previous)

        g.check(
            SignBitQuantizer.__init__.__kwdefaults__["rotation"] == previous,
            "library default restored after the gate",
            f"={previous!r}",
        )

    g.coverage(
        "k>1 is not covered. Corpus stores a single SignBitQuantizer; "
        "StackedSignBitQuantizer's k and its per-stack rotations are persisted "
        "by nothing, so a stacked index has the same silent-reinterpretation "
        "hazard this gate closes for k=1."
    )
    g.coverage(
        "LAPACK/BLAS drift is not covered. haar_rotation goes through "
        "np.linalg.qr, which is bit-deterministic on one machine but can "
        "differ across BLAS builds. Byte-identity here is asserted within a "
        "single process; a corpus written on one machine and read on another "
        "could still decode differently, and no sidecar fixes that."
    )
    g.coverage(
        "Old readers are not covered. A remax build predating rotation.json "
        "ignores the file entirely, so it will mis-decode an rht-built corpus. "
        "The sidecar protects existing indexes from a future default flip; it "
        "cannot protect past code from a future index."
    )
    g.coverage(
        "dtype is not covered. The f32/f64 working precision is likewise "
        "unrecorded, and the reader still resolves it from a constructor "
        "argument rather than from disk."
    )
    g.coverage(
        "Corrupt-sidecar handling and __repr__ are not covered. Mutation "
        "testing (mutate.py over src/remax/corpus.py, 120 sites, 57 killed / "
        "63 survived) left exactly two survivors inside the code this gate is "
        "about: the `or` joining the two malformed-payload conditions in "
        "_read_rotation, and the `==` choosing whether __repr__ prints the "
        "rotation. Both are message and diagnostics paths, not decode paths, "
        "and both are pinned by tests/test_corpus.py instead. The other 61 "
        "survivors are outside the rotation path entirely — header "
        "validation, the legacy v0 reader, SQLite chunking, centering — which "
        "this gate deliberately does not exercise. (Run mutate.py with "
        "PYTHONDONTWRITEBYTECODE=1: same-length mutations rewritten inside one "
        "second reuse a stale .pyc and report as false survivors.)"
    )

    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
