"""Shared cache-path resolution for the bench/*nemotron*.py family.

Every one of those scripts needs the same three directories, and each one
used to hardcode them. Four scripts, three different conventions:

* ``eval_nemotron_1bit.py`` read ``NEMOTRON_SCRATCH`` (plus
  ``NEMOTRON_DATA_DIR`` / ``NEMOTRON_EMB_DIR``) — the good one.
* ``embed_nemotron.py`` read ``SCRATCH``, a different variable, so setting
  one did not set the other.
* ``baselines_nemotron.py`` and ``latency_nemotron.py`` read nothing at all.
  Their paths could only be changed by editing the file.

Worse, all four defaulted to the *same literal absolute path*::

    /tmp/claude-0/-home-user/17f19a5d-.../scratchpad

which is a scratch directory belonging to a session that ended long ago. It
does not exist on any machine now. So the default was not "the usual place",
it was "a path that cannot resolve", and two of the four scripts offered no
way to say otherwise.

Resolution order, applied identically everywhere:

1. ``NEMOTRON_SCRATCH``
2. ``SCRATCH`` (kept so anything already exporting it keeps working)
3. ``<repo>/bench/.cache/nemotron`` — inside the repo, already gitignored,
   and it exists on whatever machine is running.

``NEMOTRON_DATA_DIR`` and ``NEMOTRON_EMB_DIR`` still override the two
subdirectories individually, so a cache unpacked by
``fetch_nemotron_cache.sh`` can be pointed at without moving anything.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DATA_DIR", "EMB_DIR", "RESULTS_DIR", "SCRATCH", "describe"]

_BENCH_DIR = Path(__file__).resolve().parent
_DEFAULT_SCRATCH = _BENCH_DIR / ".cache" / "nemotron"

SCRATCH = Path(
    os.environ.get("NEMOTRON_SCRATCH")
    or os.environ.get("SCRATCH")
    or _DEFAULT_SCRATCH
)
DATA_DIR = Path(os.environ.get("NEMOTRON_DATA_DIR") or SCRATCH / "data")
EMB_DIR = Path(os.environ.get("NEMOTRON_EMB_DIR") or SCRATCH / "emb")
RESULTS_DIR = _BENCH_DIR / "results"


def describe() -> str:
    """One-line summary for a script banner, so a run says where it looked."""
    return (
        f"scratch={SCRATCH}  data={DATA_DIR}  emb={EMB_DIR}\n"
        f"  (override with NEMOTRON_SCRATCH, or NEMOTRON_DATA_DIR / "
        f"NEMOTRON_EMB_DIR individually)"
    )
