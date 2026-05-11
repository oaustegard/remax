"""BEIR / NFCorpus BM25 sketch bench — ``python bench/run_bm25_sketch.py``.

Thin shim around :mod:`remax.bench.bm25_sketch`. The real implementation
lives in the installed package so it is importable from tests; this
script exists so that the issue #36 invocation contract
(``python bench/run_bm25_sketch.py``) works directly out of a source
checkout.

Usage
-----
::

    # one-time: fetch NFCorpus into bench/.cache/NFCorpus/ (~30 MB)
    bash bench/fetch_nfcorpus.sh

    # produce bench/results/BM25_SKETCH.md
    python bench/run_bm25_sketch.py

See ``python bench/run_bm25_sketch.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.bm25_sketch import main


if __name__ == "__main__":
    raise SystemExit(main())
