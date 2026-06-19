"""Issue #46 ITQ spike harness — `python bench/run_itq_experiment.py`.

Thin shim around :mod:`remax.bench.run_itq`. The real implementation lives in
the installed package so it is importable from tests; this script exists so the
experiment runs directly out of a source checkout.

Usage
-----
::

    # one-time: fetch the broad SPECTER2 cache (≈30 MB)
    bash bench/fetch_specter2_cache.sh

    # plus the narrow corpus for the cross-corpus transfer probe, into
    #   bench/.cache/SPECTER2_NARROW/embeddings.npy
    # (specter2_nlp_narrow.npy from
    #  oaustegard/claude-container-layers@specter2-nlp-narrow-10k)

    # produce bench/results/ITQ.md
    python bench/run_itq_experiment.py

    # smaller smoke run
    python bench/run_itq_experiment.py --n 2000 --queries 50 --itq-iters 20

See ``python bench/run_itq_experiment.py --help`` for the full flag set.
"""

from __future__ import annotations

from remax.bench.run_itq import main


if __name__ == "__main__":
    raise SystemExit(main())
