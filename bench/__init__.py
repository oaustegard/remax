"""bench — remax's evaluation harness. Not part of the installed library.

Submodules:

* :mod:`bench.eval` — recall@K and float32 ground-truth helpers.
* :mod:`bench.datasets` — cached real-embedding loaders (SPECTER2,
  MiniLM-L6-v2, GloVe-300d).
* :mod:`bench.run_baseline` — CLI driver that produces ``BASELINE.md``.
* :mod:`bench.run_rerank` / :mod:`bench.run_topn_sweep` — stage-2 rerank
  experiments; :mod:`bench.rerank` holds the rerankers they drive.
* :mod:`bench.crossover` — remax vs remex at matched bit budgets.

Everything else in this directory is a standalone script: run it, don't
import it.

Why this is a package at all
----------------------------
These six modules import each other and are imported by ``tests/``, so they
need a stable import path. That used to be ``remax.bench``, i.e. **inside the
installed wheel** — the stated reason being that ``setuptools.packages.find``
was configured for ``src/`` only, so a top-level ``bench/`` could not be
imported. That was a packaging assumption, not a fact: pytest puts the repo
root on ``sys.path``, so ``bench`` imports from a checkout without being
installed, and ``where = ["src"]`` keeps it out of the wheel for exactly the
same reason it was thought to be the obstacle.

Moved out of ``src/remax/`` on 2026-08-03. Downstreams that install ``remax``
no longer receive ~3,100 lines of benchmark driver, and the four one-line
shims at the repo root that re-exposed it are gone with it.

The four CLI modules run both ways — ``python -m bench.run_baseline`` and
``python bench/run_baseline.py`` — because the second form is the invocation
contract written into the issues and into ``bench/results/*.md``.
"""

from __future__ import annotations

__all__ = ["crossover", "datasets", "eval", "rerank", "run_baseline",
           "run_rerank", "run_topn_sweep"]
