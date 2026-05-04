"""Top-N sweep driver — `python bench/run_topn_sweep.py`.

Thin shim around :mod:`remax.bench.run_topn_sweep`. The real
implementation lives in the installed package so it is importable from
tests; this script exists so that ``python bench/run_topn_sweep.py``
works directly out of a source checkout.

Usage
-----
::

    # one-time: fetch the SPECTER2 cache (embeddings + texts)
    bash bench/fetch_specter2_cache.sh

    # produce bench/results/{rerank_topn_sweep.csv,.png,RERANK_topn_sweep.md}
    python bench/run_topn_sweep.py

    # plateau probe without the cross-encoder (fast)
    python bench/run_topn_sweep.py --no-cross-encoder

    # smaller smoke run
    python bench/run_topn_sweep.py --queries 20 --top-n 50,100,200

    # compare an off-the-shelf MiniLM CE against a domain-matched one
    python bench/run_topn_sweep.py \\
        --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --cross-encoder-model your-org/scibert-msmarco-onnx

See ``python bench/run_topn_sweep.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.run_topn_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
