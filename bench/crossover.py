"""Crossover plot driver — `python bench/crossover.py`.

Thin shim around :mod:`remax.bench.crossover`. The real implementation
lives in the installed package so it is importable from tests; this script
exists so the issue #5 invocation contract (``python bench/crossover.py``)
works directly out of a source checkout.

Usage
-----
::

    # one-time: fetch the SPECTER2 cache (~30 MB)
    bash bench/fetch_specter2_cache.sh

    # produce bench/results/{crossover.csv, crossover.png, CROSSOVER.md}
    python bench/crossover.py

    # subset / smaller smoke runs
    python bench/crossover.py --datasets SPECTER2 --n 1000 --queries 50

See ``python bench/crossover.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.crossover import main


if __name__ == "__main__":
    raise SystemExit(main())
