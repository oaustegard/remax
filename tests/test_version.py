"""The version must come from package metadata, not a hand-edited literal.

Why this test exists: ``remax.__version__`` sat at ``"0.0.0"`` through 35+
merged pull requests, including PR #31 (f64 -> f32 default), which *changes
the emitted codes*. Downstream, ``remax_kb`` stamps ``remax_version`` into
every ``.kb`` / ``.kbi`` manifest, and ``SPEC_v2`` promises that the same
``(corpus, embedder, dim, k, seed)`` yields a bit-identical ``vectors.bin``.
A constant version means the one field able to *detect* a violation of that
promise could never do so.

The literal check runs whether or not the package is installed, so the guard
is not inert in exactly the checkout where a regression would be introduced.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

import remax

_INIT = pathlib.Path(remax.__file__)
_PYPROJECT = _INIT.parents[2] / "pyproject.toml"


def test_version_is_not_a_hardcoded_literal() -> None:
    """``__init__`` must not assign a string literal to ``__version__``.

    Runs from source, so it fires in the working tree where such an edit
    would land -- installed or not.
    """
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    assignments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "__version__"
    ]
    assert assignments, "__init__.py never assigns __version__"

    # A literal fallback inside the `except PackageNotFoundError` branch is
    # legitimate -- it is what a bare source checkout gets. What must not
    # happen is that EVERY assignment is a literal, which is the state where
    # the version stops tracking releases.
    assert any(not isinstance(value, ast.Constant) for value in assignments), (
        "every __version__ assignment in __init__.py is a string literal; it "
        "must be resolved from package metadata (importlib.metadata.version)."
    )


def test_version_is_resolvable() -> None:
    assert remax.__version__
    assert remax.__version__ != "0.0.0", (
        "__version__ resolved to the sentinel that shipped in every .kb manifest "
        "for 35+ PRs -- the package is probably not installed."
    )


@pytest.mark.skipif(not _PYPROJECT.exists(), reason="not a source checkout")
def test_version_matches_pyproject() -> None:
    """Metadata and pyproject must agree, so a release cannot ship skewed."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version field in pyproject.toml"
    declared = match.group(1)

    if remax.__version__.endswith("+unknown"):
        pytest.skip("remax is not installed; metadata unavailable")
    assert remax.__version__ == declared, (
        f"installed metadata says {remax.__version__!r} but pyproject declares "
        f"{declared!r} -- reinstall, or the release is skewed."
    )
