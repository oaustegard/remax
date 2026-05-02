"""Real-embedding baseline harness — `python bench/run_baseline.py`.

Thin shim around :mod:`remax.bench.run_baseline`. The real implementation
lives in the installed package so it is importable from tests; this script
exists so that the issue #4 invocation contract (``python bench/run_baseline.py``)
works directly out of a source checkout.

Usage
-----
::

    # one-time: fetch the SPECTER2 cache (≈30 MB)
    bash bench/fetch_specter2_cache.sh

    # produce bench/results/BASELINE.md
    python bench/run_baseline.py

    # subset / smaller smoke runs
    python bench/run_baseline.py --datasets SPECTER2 --n 1000 --queries 50

See ``python bench/run_baseline.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.run_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
