"""Every in-repo path a Markdown file points at must resolve.

Why this file exists
--------------------
``bench/results/LFM25_SUMMARY.md`` cited two files that have never existed in
this repository's history (verified against the full log, not a shallow
clone): ``LFM25_ASYMMETRIC.md``, cited from inside a *correction* block — so
the pointer meant to substantiate a retraction went nowhere — and
``docs/research/matryoshka-and-quantization.md``, cited as "the standing
conclusion", from a ``docs/research/`` directory that does not exist either.

Both read as authoritative. A named file with a plausible path is one of the
most convincing things prose can contain, and nothing about it looks wrong.
Nothing in a repo with no CI was ever going to notice.

What is checked, and what is not
--------------------------------
Two classes of reference, deliberately treated differently:

**Markdown links** ``[text](target)`` — always checked. A link is an explicit
promise of navigability, and there is no ambiguity about intent.

**Backticked paths** ``` `some/path.md` ``` — checked only when they contain a
``/`` (so they are unmistakably paths), or when they are a bare ``*.md``
filename resolvable next to the citing document. A bare ``core.py`` or
``geometry.py`` in prose is a *name*, not a path — it means "the core module",
and demanding it resolve from the citing file's directory would produce noise
that trains people to ignore this test. Both dangling references above are
caught under these rules: one has a ``/``, one is a bare ``.md``.

Anything genuinely outside the repo goes in ``KNOWN_EXTERNAL`` with a written
reason. An allowlist with reasons stays auditable; a silent skip does not.

Cannot catch: links whose *target exists but is wrong* (pointing at the right
kind of file with the wrong content); URLs, which are not fetched; bare
non-``.md`` filenames, per the rule above; anchors within a document — the
fragment is stripped before resolving, so ``FILE.md#no-such-heading`` passes
as long as ``FILE.md`` exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Refs that legitimately do not resolve inside this repo. Reason required —
#: an entry without one is indistinguishable from a defect someone silenced.
KNOWN_EXTERNAL: dict[str, str] = {
    # CLAUDE.md points at the remex repo for comparison implementations.
    "remex/codebook.py": "path in oaustegard/remex, cited for contrast",
    "remex/core.py": "path in oaustegard/remex, cited for contrast",
    "bench/specter2_eval.py": "path in oaustegard/remex, cited as a harness to port",
    "bench/onebit_experiment.py": "path in oaustegard/remex, cited as a harness to port",
    # The sparse path was built, measured, found not to work, and removed.
    # BM25_SKETCH.md is the negative result and names what it removed.
    "src/remax/bm25.py": "removed with the sparse path; BM25_SKETCH.md is its negative result",
    "src/remax/sparse.py": "removed with the sparse path; BM25_SKETCH.md is its negative result",
}

_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*([^)\s]+)")
_CODE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|py|csv|png|json|toml|sh|txt|yml|yaml))`")
_FENCE = re.compile(r"^[ ]{0,3}```.*?^[ ]{0,3}```", re.MULTILINE | re.DOTALL)


def _markdown_files() -> list[Path]:
    return sorted(
        p for p in REPO.rglob("*.md")
        if ".git" not in p.parts and "node_modules" not in p.parts
    )


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#", "<"))


def _references(path: Path) -> set[tuple[str, str]]:
    """Return ``(kind, target)`` pairs worth resolving, for one file."""
    text = path.read_text(encoding="utf-8")
    refs: set[tuple[str, str]] = set()

    for m in _LINK.finditer(text):
        target = m.group(1)
        if _is_external(target):
            continue
        refs.add(("link", target.split("#")[0]))

    # Backticked paths, scanned with links and fenced code removed: link text
    # is frequently a backticked filename ([`BASELINE.md`](bench/results/...)),
    # and counting that as a bare reference would resolve it in the wrong
    # directory and fail on a link that is perfectly correct.
    prose = _FENCE.sub("", text)
    prose = _LINK.sub("", prose)
    for m in _CODE.finditer(prose):
        target = m.group(1)
        if "/" in target or target.endswith(".md"):
            refs.add(("code", target))

    return refs


@pytest.mark.parametrize(
    "md", _markdown_files(), ids=lambda p: str(p.relative_to(REPO))
)
def test_markdown_references_resolve(md: Path) -> None:
    dangling = []
    for kind, target in sorted(_references(md)):
        if target in KNOWN_EXTERNAL:
            continue
        if (md.parent / target).exists() or (REPO / target).exists():
            continue
        dangling.append(f"  {kind}: {target}")
    assert not dangling, (
        f"{md.relative_to(REPO)} references paths that do not exist:\n"
        + "\n".join(dangling)
        + "\n(if a reference is genuinely outside this repo, add it to "
          "KNOWN_EXTERNAL in this file with a reason)"
    )


def test_scanner_finds_references() -> None:
    """Guard the scanner: a regex that stops matching would pass everything."""
    files = _markdown_files()
    assert len(files) >= 5, f"expected the repo's markdown corpus, found {files}"
    total = sum(len(_references(p)) for p in files)
    assert total >= 20, (
        f"only {total} in-repo references extracted across {len(files)} files; "
        "the extraction regexes have probably stopped matching"
    )


def test_known_external_entries_are_still_needed() -> None:
    """An allowlist nobody prunes silently accumulates real defects.

    If a path in KNOWN_EXTERNAL now exists in the repo, the exemption is
    stale and hiding whatever that file does or does not contain.
    """
    stale = [t for t in KNOWN_EXTERNAL if (REPO / t).exists()]
    assert not stale, (
        f"KNOWN_EXTERNAL exempts paths that now exist in the repo: {stale}. "
        "Remove the entries so the references are checked again."
    )


def test_known_external_entries_are_actually_referenced() -> None:
    """The other direction: an exemption for a reference nobody makes any more."""
    referenced = set()
    for p in _markdown_files():
        referenced |= {t for _, t in _references(p)}
    unused = sorted(set(KNOWN_EXTERNAL) - referenced)
    assert not unused, (
        f"KNOWN_EXTERNAL carries entries nothing references: {unused}. "
        "Delete them; a stale exemption is a hole with a green light on it."
    )
