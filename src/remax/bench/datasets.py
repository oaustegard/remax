"""Cached real-embedding loaders for the v0.1.0 baseline.

The bench harness expects precomputed embeddings under
``<_CACHE_ROOT>/<NAME>/embeddings.npy``. This decouples the harness from any
particular encoder library (sentence-transformers, gensim, transformers) so
that running the baseline does not require installing PyTorch.

Caches are produced by:

* SPECTER2 — ``bench/fetch_specter2_cache.sh`` reuses the publication cache
  hosted on ``oaustegard/claude-container-layers`` releases (see remex's
  fetcher), then symlinks/copies the ``.npy`` into ``SPECTER2/embeddings.npy``.
* MiniLM-L6-v2 — generate from any text corpus with sentence-transformers,
  save with ``np.save``.
* GloVe-300d — convert the standard GloVe-840B-300d text format with
  ``gensim`` and ``np.save``.

If a cache is missing, :func:`load_dataset` raises :class:`FileNotFoundError`
with a remediation hint pointing at the fetcher script.

Some datasets also ship the source text (titles + abstracts for SPECTER2),
needed by stage-2 cross-encoder rerank experiments. These live alongside the
embeddings as ``<_CACHE_ROOT>/<NAME>/texts.json`` and are loaded with
:func:`load_texts`. Datasets without a registered text cache raise
:class:`ValueError` from :func:`texts_path` / :func:`load_texts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DatasetSpec",
    "available_datasets",
    "dataset_spec",
    "dataset_path",
    "load_dataset",
    "texts_path",
    "load_texts",
]


def _default_cache_root() -> Path:
    """Locate the in-repo ``bench/.cache/`` directory.

    The package source lives at ``<repo>/src/remax/bench/datasets.py`` in an
    editable install, so the repo root is ``parents[3]``. If that path does
    not look like a remax checkout (e.g. in a wheel install on PyPI), fall
    back to a package-local ``.cache/`` so loads still work, just with a
    different conventional location.

    Override with the ``REMAX_BENCH_CACHE_DIR`` environment variable for
    out-of-tree caches.
    """
    import os

    env = os.environ.get("REMAX_BENCH_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    # src/remax/bench/datasets.py → parents[3] = repo root
    if len(here.parents) >= 4:
        candidate = here.parents[3] / "bench" / ".cache"
        # Trust the candidate iff a sibling pyproject.toml exists; otherwise
        # we're not in a remax source checkout and the package-local fallback
        # is safer.
        if (here.parents[3] / "pyproject.toml").exists():
            return candidate
    return here.parent / ".cache"


# Default cache root. Tests monkeypatch this so they can write fake caches
# into a tmpdir without touching the user's filesystem.
_CACHE_ROOT: Path = _default_cache_root()


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a registered dataset.

    ``has_texts`` is true when a ``texts.json`` companion is published in the
    same release as the embeddings. The cross-encoder rerank experiment
    (issue #20) needs source text and only runs on text-bearing datasets.
    """

    name: str
    dim: int
    fetcher_hint: str
    has_texts: bool = False


_REGISTRY: dict[str, DatasetSpec] = {
    "SPECTER2": DatasetSpec(
        name="SPECTER2",
        dim=768,
        fetcher_hint="bash bench/fetch_specter2_cache.sh",
        has_texts=True,
    ),
    "MiniLM-L6-v2": DatasetSpec(
        name="MiniLM-L6-v2",
        dim=384,
        fetcher_hint=(
            "encode any text corpus with "
            "sentence-transformers/all-MiniLM-L6-v2 and "
            "np.save the (n,384) float32 array"
        ),
    ),
    "GloVe-300d": DatasetSpec(
        name="GloVe-300d",
        dim=300,
        fetcher_hint=(
            "convert glove.840B.300d.txt with gensim and "
            "np.save the (n,300) float32 array"
        ),
    ),
}


def available_datasets() -> Tuple[str, ...]:
    """Names registered for the v0.1.0 baseline."""
    return tuple(_REGISTRY.keys())


def dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` registered for ``name``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown dataset {name!r}; "
            f"available: {sorted(_REGISTRY.keys())!r}"
        )
    return _REGISTRY[name]


def dataset_path(name: str) -> Path:
    """Deterministic absolute path to a dataset's cached ``embeddings.npy``."""
    spec = dataset_spec(name)  # validates name
    return _CACHE_ROOT / spec.name / "embeddings.npy"


def load_dataset(
    name: str, n: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """Load (and optionally slice) the cached embeddings for ``name``.

    Parameters
    ----------
    name : str
        One of :func:`available_datasets`.
    n : int | None
        If given, return only the first ``n`` rows. Raises if the cache has
        fewer than ``n`` rows.

    Returns
    -------
    X : np.ndarray, shape (n', dim), dtype float32
    info : dict
        ``{"name": str, "dim": int, "n": int}`` describing the returned
        slice.

    Raises
    ------
    FileNotFoundError
        Cache file does not exist. The exception message names the missing
        path and the fetcher hint registered for ``name``.
    ValueError
        Unknown ``name``, dim mismatch, or ``n`` larger than the cache.
    """
    spec = dataset_spec(name)  # validates name
    path = dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {name} embeddings cache at {path}\n"
            f"to fix: {spec.fetcher_hint}"
        )

    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != spec.dim:
        raise ValueError(
            f"{name} cache at {path} has shape {arr.shape}; "
            f"expected (*, {spec.dim}). dim mismatch — re-fetch the cache."
        )
    if n is not None:
        if n > arr.shape[0]:
            raise ValueError(
                f"cache has {arr.shape[0]} rows; n={n} requested."
            )
        arr = arr[:n]

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    info = {"name": spec.name, "dim": spec.dim, "n": int(arr.shape[0])}
    return arr, info


# --------------------------------------------------------------------- #
# Texts
# --------------------------------------------------------------------- #


def texts_path(name: str) -> Path:
    """Deterministic absolute path to a dataset's cached ``texts.json``.

    Raises :class:`ValueError` if the dataset does not have a registered text
    companion (``has_texts=False``).
    """
    spec = dataset_spec(name)
    if not spec.has_texts:
        raise ValueError(
            f"dataset {name!r} has no registered texts cache; "
            f"only datasets with has_texts=True can supply source text."
        )
    return _CACHE_ROOT / spec.name / "texts.json"


def load_texts(name: str, n: Optional[int] = None) -> Tuple[List[str], dict]:
    """Load (and optionally slice) the cached source texts for ``name``.

    The texts file is a JSON list of strings, aligned by index with the
    embeddings array. For SPECTER2 each entry is ``"<title> [SEP] <abstract>"``
    as encoded into the upstream embedding.

    Parameters
    ----------
    name : str
        One of :func:`available_datasets` with ``has_texts=True``.
    n : int | None
        If given, return only the first ``n`` entries. Raises if the cache
        has fewer than ``n`` entries.

    Returns
    -------
    texts : list[str]
    info : dict
        ``{"name": str, "n": int}``.

    Raises
    ------
    FileNotFoundError
        Texts cache does not exist. Message names the missing path and the
        fetcher hint registered for ``name``.
    ValueError
        Unknown ``name``, dataset has no text cache, or ``n`` exceeds the
        cache.
    """
    spec = dataset_spec(name)  # validates name
    path = texts_path(name)  # validates has_texts
    if not path.exists():
        raise FileNotFoundError(
            f"missing {name} texts cache at {path}\n"
            f"to fix: {spec.fetcher_hint}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list) or not all(isinstance(t, str) for t in items):
        raise ValueError(
            f"{name} texts cache at {path} is not a JSON list of strings."
        )
    if n is not None:
        if n > len(items):
            raise ValueError(
                f"texts cache has {len(items)} entries; n={n} requested."
            )
        items = items[:n]
    info = {"name": spec.name, "n": int(len(items))}
    return items, info
