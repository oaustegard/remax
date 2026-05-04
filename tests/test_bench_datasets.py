"""Tests for ``remax.bench.datasets`` — cache-path resolution and loaders.

Real embeddings (SPECTER2, MiniLM-L6-v2, GloVe-300d) live in a per-dataset
cache under ``bench/.cache/<NAME>/embeddings.npy``. The dataset module
provides:

  * :func:`available_datasets` — names registered for the v0.1.0 baseline.
  * :func:`dataset_spec` — declared ``(name, dim)`` plus the fetcher hint
    that should appear in error messages.
  * :func:`dataset_path` — deterministic absolute path under the cache.
  * :func:`load_dataset` — read the cache, slice ``n``, return float32.

Tests use a tmpdir + a monkeypatched cache root rather than real downloads,
so they run offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from remax.bench import datasets


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


def test_available_datasets_includes_required_three():
    """v0.1.0 baseline requires SPECTER2, MiniLM-L6-v2, GloVe-300d."""
    names = datasets.available_datasets()
    assert "SPECTER2" in names
    assert "MiniLM-L6-v2" in names
    assert "GloVe-300d" in names


def test_dataset_spec_returns_dim_for_each():
    """Each registered dataset must expose its expected dimension."""
    assert datasets.dataset_spec("SPECTER2").dim == 768
    assert datasets.dataset_spec("MiniLM-L6-v2").dim == 384
    assert datasets.dataset_spec("GloVe-300d").dim == 300


def test_dataset_spec_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.dataset_spec("not-a-real-dataset")


# --------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------- #


def test_dataset_path_is_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    p = datasets.dataset_path("SPECTER2")
    assert isinstance(p, Path)
    assert tmp_path in p.parents
    # Filename must contain the dataset name to make caches debuggable
    assert "SPECTER2" in str(p)


def test_dataset_path_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    a = datasets.dataset_path("MiniLM-L6-v2")
    b = datasets.dataset_path("MiniLM-L6-v2")
    assert a == b


def test_dataset_path_unknown_name_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError):
        datasets.dataset_path("not-a-real-dataset")


# --------------------------------------------------------------------- #
# load_dataset
# --------------------------------------------------------------------- #


def _write_cache(monkeypatch, tmp_path, name, n, d, seed=0, dtype=np.float32):
    """Helper: write a fake embeddings.npy cache for ``name`` under tmp_path."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, d)).astype(dtype)
    p = datasets.dataset_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return arr


def test_load_dataset_reads_full_cache_when_n_none(monkeypatch, tmp_path):
    arr = _write_cache(monkeypatch, tmp_path, "SPECTER2", n=50, d=768)
    X, info = datasets.load_dataset("SPECTER2")
    assert X.shape == (50, 768)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, arr)
    assert info["name"] == "SPECTER2"
    assert info["dim"] == 768


def test_load_dataset_slices_to_n(monkeypatch, tmp_path):
    arr = _write_cache(monkeypatch, tmp_path, "SPECTER2", n=100, d=768)
    X, info = datasets.load_dataset("SPECTER2", n=30)
    assert X.shape == (30, 768)
    np.testing.assert_array_equal(X, arr[:30])


def test_load_dataset_promotes_dtype_to_float32(monkeypatch, tmp_path):
    """Cache may be written as float64 by some loaders; we want float32 out."""
    _write_cache(
        monkeypatch, tmp_path, "MiniLM-L6-v2", n=10, d=384, dtype=np.float64
    )
    X, _ = datasets.load_dataset("MiniLM-L6-v2")
    assert X.dtype == np.float32


def test_load_dataset_dim_mismatch_raises(monkeypatch, tmp_path):
    """Cache file with wrong second axis must fail loudly."""
    _write_cache(monkeypatch, tmp_path, "SPECTER2", n=10, d=64)  # wrong dim
    with pytest.raises(ValueError, match="dim"):
        datasets.load_dataset("SPECTER2")


def test_load_dataset_n_too_large_raises(monkeypatch, tmp_path):
    _write_cache(monkeypatch, tmp_path, "SPECTER2", n=10, d=768)
    with pytest.raises(ValueError, match="cache has"):
        datasets.load_dataset("SPECTER2", n=100)


def test_load_dataset_missing_cache_message_includes_fetcher(
    monkeypatch, tmp_path
):
    """Missing cache → FileNotFoundError whose message tells the user how to fix
    it. The fetcher hint is the contract — without it, the user is stuck."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        datasets.load_dataset("SPECTER2")
    msg = str(excinfo.value)
    # Path of the missing file appears in the error
    assert "SPECTER2" in msg
    # And so does the fetcher script the user is expected to run
    assert "fetch_specter2_cache" in msg or "fetch" in msg.lower()


def test_load_dataset_unknown_name_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError):
        datasets.load_dataset("not-a-real-dataset")


# --------------------------------------------------------------------- #
# texts_path / load_texts
# --------------------------------------------------------------------- #


def _write_texts(monkeypatch, tmp_path, name, items):
    """Helper: write a fake texts.json cache for ``name`` under tmp_path."""
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    p = datasets.texts_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items), encoding="utf-8")
    return items


def test_specter2_spec_advertises_texts():
    """SPECTER2 ships a texts companion; the registry must expose that."""
    spec = datasets.dataset_spec("SPECTER2")
    assert spec.has_texts is True


def test_glove_and_minilm_have_no_texts():
    """The other datasets in the registry don't ship text in v0.1.0."""
    assert datasets.dataset_spec("MiniLM-L6-v2").has_texts is False
    assert datasets.dataset_spec("GloVe-300d").has_texts is False


def test_texts_path_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    p = datasets.texts_path("SPECTER2")
    assert isinstance(p, Path)
    assert tmp_path in p.parents
    assert p.name == "texts.json"
    assert "SPECTER2" in str(p)


def test_texts_path_raises_for_dataset_without_texts(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="no registered texts"):
        datasets.texts_path("MiniLM-L6-v2")


def test_load_texts_reads_full_cache_when_n_none(monkeypatch, tmp_path):
    items = [f"doc {i}" for i in range(7)]
    _write_texts(monkeypatch, tmp_path, "SPECTER2", items)
    out, info = datasets.load_texts("SPECTER2")
    assert out == items
    assert info == {"name": "SPECTER2", "n": 7}


def test_load_texts_slices_to_n(monkeypatch, tmp_path):
    items = [f"doc {i}" for i in range(20)]
    _write_texts(monkeypatch, tmp_path, "SPECTER2", items)
    out, info = datasets.load_texts("SPECTER2", n=5)
    assert out == items[:5]
    assert info["n"] == 5


def test_load_texts_n_too_large_raises(monkeypatch, tmp_path):
    _write_texts(monkeypatch, tmp_path, "SPECTER2", ["a", "b", "c"])
    with pytest.raises(ValueError, match="texts cache has"):
        datasets.load_texts("SPECTER2", n=10)


def test_load_texts_missing_cache_message_includes_fetcher(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        datasets.load_texts("SPECTER2")
    msg = str(excinfo.value)
    assert "SPECTER2" in msg
    assert "fetch" in msg.lower()


def test_load_texts_rejects_non_string_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    p = datasets.texts_path("SPECTER2")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"title": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="list of strings"):
        datasets.load_texts("SPECTER2")


def test_load_texts_unsupported_dataset_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="no registered texts"):
        datasets.load_texts("MiniLM-L6-v2")
