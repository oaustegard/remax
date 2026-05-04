"""Stage-2 rerank experiment — `python bench/run_rerank.py`.

Thin shim around :mod:`remax.bench.run_rerank`. The real implementation
lives in the installed package so it is importable from tests; this script
exists so that ``python bench/run_rerank.py`` works directly out of a source
checkout.

Usage
-----
::

    # one-time: fetch the SPECTER2 cache (embeddings + texts)
    bash bench/fetch_specter2_cache.sh

    # produce bench/results/RERANK.md
    python bench/run_rerank.py

    # smaller smoke run
    python bench/run_rerank.py --queries 20 --top-n 50

See ``python bench/run_rerank.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.run_rerank import main


if __name__ == "__main__":
    raise SystemExit(main())
