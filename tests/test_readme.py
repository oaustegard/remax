"""Execute the README's python blocks, so documented code cannot rot.

Why this file exists
--------------------
Every ``python`` example in README.md was broken at once, and had been for
long enough that nobody could say when it broke:

* ``sq.hamming_distances(...)`` — not a method on any quantizer;
  ``hamming_distances`` is module-level in ``remax.packing``.
* ``remax.Corpus.create(...)`` — no such constructor. The real one is
  ``Corpus.build(path, vectors, ids, ...)``, it wants a *directory* rather
  than a ``.bin`` file, and it takes no quantizer argument.
* ``remax.hamming_distances(q.encode(query), codes)`` then ``dists[0]`` —
  arguments reversed (the signature is ``(codes, query_code)``) and the
  return is 1-D, so ``dists[0]`` is a scalar distance, not a row.

Every one of those is the kind of defect a human reader glides over: the
names are plausible, the shapes are plausible, and prose does not run. So
the fix is not "edit the README" — a re-edit rots the same way. The fix is
that the README is now *executed*, and a divergence between the documented
API and the real one is a test failure.

Contract
--------
Blocks are executed **in document order into one shared namespace**, so a
later block may use names an earlier one bound — that is how a reader reads
them. They run with the working directory set to a per-test temp dir, so an
example may write relative paths (``Corpus.build("papers/", ...)``) without
littering the repo or needing tempfile noise in the docs.

To exclude a block that genuinely cannot run (a snippet that needs a GPU,
credentials, or a 4 GB download), put this immediately above its fence::

    <!-- readme-exec: skip -->

Skipping is deliberately ugly and deliberately visible in the rendered
source, because every skipped block is a block that can silently go stale.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

#: ```python … ``` — the info string must be exactly ``python`` (``pycon``,
#: ``text`` and ``bash`` blocks are not executable and are not claimed to be).
_FENCE = re.compile(
    r"^(?P<indent>[ ]{0,3})```[ \t]*python[ \t]*$\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

_SKIP = re.compile(r"<!--\s*readme-exec:\s*skip\s*-->\s*$")


def _blocks(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, source)`` for each executable python block."""
    out = []
    for m in _FENCE.finditer(text):
        preceding = text[: m.start()].rstrip("\n").rsplit("\n", 1)[-1]
        if _SKIP.search(preceding):
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((line_no, m.group("body")))
    return out


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists(readme_text: str) -> None:
    assert readme_text.strip(), "README.md is empty"


def test_extraction_finds_blocks(readme_text: str) -> None:
    """Guard the harness itself.

    Without this, a regex that silently stops matching turns the whole file
    into a test that executes zero blocks and reports green — an assertion
    whose truth no longer depends on the subject.
    """
    blocks = _blocks(readme_text)
    assert blocks, (
        "no executable ```python blocks found in README.md — either the "
        "examples were deleted or the fence regex stopped matching"
    )


def test_readme_python_blocks_execute(readme_text: str, tmp_path: Path) -> None:
    """Run every documented example, in order, in one namespace."""
    blocks = _blocks(readme_text)
    assert blocks  # covered by its own test; restated so failures read clearly

    namespace: dict = {"__name__": "__readme__", "__file__": str(README)}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for line_no, source in blocks:
            code = compile(source, f"{README}:{line_no}", "exec")
            try:
                exec(code, namespace)  # noqa: S102 - executing docs is the point
            except Exception as exc:  # pragma: no cover - the failure path
                raise AssertionError(
                    f"README.md block starting at line {line_no} failed to "
                    f"execute: {type(exc).__name__}: {exc}\n"
                    f"--- block ---\n{source}"
                ) from exc
    finally:
        os.chdir(cwd)


def test_readme_examples_bind_the_documented_names(
    readme_text: str, tmp_path: Path
) -> None:
    """Executing is necessary but not sufficient — check what it produced.

    A block can run cleanly and still document nothing: ``import remax`` on
    its own executes fine. These assertions pin the shapes the prose claims,
    so an example that runs but has drifted from the API still fails.
    """
    namespace: dict = {"__name__": "__readme__", "__file__": str(README)}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for line_no, source in _blocks(readme_text):
            exec(compile(source, f"{README}:{line_no}", "exec"), namespace)
    finally:
        os.chdir(cwd)

    import numpy as np

    import remax

    codes = namespace["codes"]
    assert isinstance(codes, np.ndarray) and codes.dtype == np.uint8
    n, d = namespace["embeddings"].shape
    assert codes.shape == (n, d // 8), "the (n, d//8) claim in the comment"

    # hamming_distances is 1-D over the corpus, and the comment says so.
    all_dists = namespace["all_dists"]
    assert all_dists.shape == (n,)

    # k stacked rotations = k bytes per 8 dims.
    stacked = namespace["stacked_codes"]
    assert stacked.shape == (n, 4 * d // 8)

    # Corpus.build returns a Corpus, and search resolves metadata.
    corpus = namespace["corpus"]
    assert isinstance(corpus, remax.Corpus)
    results = namespace["results"]
    assert len(results) == 10
    assert [r.rank for r in results] == list(range(10))
    assert all(isinstance(r, remax.Result) for r in results)
    assert all(r.record_id.startswith("paper-") for r in results)
    assert all(isinstance(r.meta, dict) and "title" in r.meta for r in results)


def test_readme_does_not_reference_a_nonexistent_api(readme_text: str) -> None:
    """Catch the *shape* of the original defect, not just the instances.

    The three broken calls all had the same form: a name that reads like the
    API and is not in it. Executing the blocks catches that only where the
    line is reached; this scans the prose too, where nothing executes.
    """
    import remax

    referenced = set(re.findall(r"\bremax\.([A-Za-z_][A-Za-z0-9_]*)", readme_text))
    missing = sorted(name for name in referenced if not hasattr(remax, name))
    assert not missing, f"README references remax.{{{', '.join(missing)}}}, which do not exist"

    # Attribute access on the objects the README constructs, e.g. the
    # `sq.hamming_distances(...)` that started this.
    for owner, attrs in (
        (remax.SignBitQuantizer, re.findall(r"\bq\.([A-Za-z_]\w*)\(", readme_text)),
        (
            remax.StackedSignBitQuantizer,
            re.findall(r"\bsq\.([A-Za-z_]\w*)\(", readme_text),
        ),
        (remax.Corpus, re.findall(r"\bcorpus\.([A-Za-z_]\w*)\(", readme_text)),
    ):
        for attr in set(attrs):
            assert hasattr(owner, attr), (
                f"README calls {owner.__name__}.{attr}(), which does not exist"
            )
