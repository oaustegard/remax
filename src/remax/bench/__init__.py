"""remax.bench — real-embedding evaluation harness for v0.1.0.

Submodules:

* :mod:`remax.bench.eval` — recall@K and float32 ground-truth helpers.
* :mod:`remax.bench.datasets` — cached real-embedding loaders (SPECTER2,
  MiniLM-L6-v2, GloVe-300d).
* :mod:`remax.bench.run_baseline` — CLI driver that produces ``BASELINE.md``.

This package lives inside the installed ``remax`` distribution because the
in-repo ``bench/`` directory was tarpitted by ``setuptools.packages.find``
under ``src/`` only. Source-of-truth is ``src/remax/bench/``; the legacy
``bench/`` directory at repo root holds smoke scripts, a fetcher, and the
``results/`` artifacts.
"""

from __future__ import annotations

__all__ = ["datasets", "eval", "run_baseline"]
