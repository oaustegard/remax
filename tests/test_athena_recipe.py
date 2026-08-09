"""Execute ``docs/athena-recipe.md`` and check the SQL it generates ranks right.

Why this file exists
--------------------
The Athena recipe asks a reader to reimplement remax's Hamming scan in a
second place — as ``bit_count(bitwise_xor(...), 64)`` over int64 limb columns
— and the two implementations must agree bit for bit. That is a much worse
failure mode than a broken example. A mistake in the byte-to-limb transform
does not raise: the query still returns 100 rows, still ranked by ascending
distance, just distance to a vector nobody asked about. Nothing about the
output looks wrong.

So the recipe is executed rather than trusted. ``to_limbs`` and
``hamming_sql`` are extracted from the document, the SQL string they produce
is parsed back apart, and its terms are evaluated under Trino's semantics
against a corpus that ``Corpus.search`` also scans. The two rankings must be
identical.

What is and is not covered
--------------------------
Covered: the limb transform, the generated SQL expression, tie ordering, the
zero-padding path for codes whose byte length is not a multiple of 8, and
stacked codes.

Not covered: anything requiring AWS or pyarrow. Those blocks carry an
``<!-- athena-exec: skip -->`` marker, following ``tests/test_readme.py``'s
convention — deliberately visible, because a skipped block can go stale.
Notably the Parquet writer is unexecuted; what it writes is ``to_limbs``
output, which is checked here, but the pyarrow call itself is not.

Also not covered, and not coverable from here: that Athena accepts this SQL.
The function signatures come from the Trino reference. See the recipe's
closing section.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from remax import Corpus, SignBitQuantizer, StackedSignBitQuantizer

DOC = Path(__file__).resolve().parent.parent / "docs" / "athena-recipe.md"

#: The block that becomes the ``remax_athena`` module the other blocks import.
#: Keyed on its leading comment so the doc names the file and the test agrees.
_MODULE_MARKER = "# remax_athena.py"

_FENCE = re.compile(
    r"^(?P<indent>[ ]{0,3})```[ \t]*python[ \t]*$\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_SKIP = re.compile(r"<!--\s*athena-exec:\s*skip\s*-->\s*$")

#: One summand of the generated distance expression.
_TERM = re.compile(r"bit_count\(bitwise_xor\((c\d+), (-?\d+)\), 64\)")


def _blocks(text: str) -> list[tuple[int, str]]:
    out = []
    for m in _FENCE.finditer(text):
        preceding = text[: m.start()].rstrip("\n").rsplit("\n", 1)[-1]
        if _SKIP.search(preceding):
            continue
        out.append((text.count("\n", 0, m.start()) + 1, m.group("body")))
    return out


@pytest.fixture(scope="module")
def recipe() -> dict:
    """Namespace holding every executable name the recipe defines.

    The ``remax_athena.py`` block is installed into ``sys.modules`` under that
    name first, so the later blocks' ``from remax_athena import to_limbs``
    resolves to the code printed in the document rather than to a copy kept
    here — which would defeat the point.
    """
    text = DOC.read_text(encoding="utf-8")
    blocks = _blocks(text)
    assert blocks, (
        "no executable ```python blocks in docs/athena-recipe.md — either "
        "they were deleted or the fence regex stopped matching"
    )

    module_src = [(ln, src) for ln, src in blocks
                  if src.lstrip().startswith(_MODULE_MARKER)]
    assert len(module_src) == 1, (
        f"expected exactly one block starting with {_MODULE_MARKER!r}, "
        f"found {len(module_src)}"
    )
    (mod_line, mod_body), = module_src

    module = types.ModuleType("remax_athena")
    exec(compile(mod_body, f"{DOC}:{mod_line}", "exec"), module.__dict__)
    sys.modules["remax_athena"] = module
    try:
        ns: dict = {"__name__": "__athena_recipe__", "__file__": str(DOC)}
        for line_no, source in blocks:
            if source is mod_body:
                continue
            try:
                exec(compile(source, f"{DOC}:{line_no}", "exec"), ns)
            except Exception as exc:  # pragma: no cover - the failure path
                raise AssertionError(
                    f"docs/athena-recipe.md block at line {line_no} failed: "
                    f"{type(exc).__name__}: {exc}\n--- block ---\n{source}"
                ) from exc
    finally:
        sys.modules.pop("remax_athena", None)

    ns["to_limbs"] = module.to_limbs
    return ns


def _trino_distances(limbs: np.ndarray, sql: str) -> np.ndarray:
    """Evaluate the generated expression the way Trino would.

    ``bit_count(x, 64)`` counts set bits in the two's-complement 64-bit
    representation, so the XOR result is popcounted as unsigned.
    """
    terms = _TERM.findall(sql)
    assert terms, f"no bit_count terms found in generated SQL:\n{sql}"
    assert len(terms) == limbs.shape[1], (
        f"{len(terms)} SQL terms for {limbs.shape[1]} limb columns"
    )
    total = np.zeros(limbs.shape[0], dtype=np.int64)
    for j, (col, literal) in enumerate(terms):
        assert col == f"c{j}", f"term {j} reads {col}, expected c{j}"
        x = np.bitwise_xor(limbs[:, j], np.int64(literal)).view(np.uint64)
        total += np.array([bin(int(v)).count("1") for v in x], dtype=np.int64)
    return total


# ---------------------------------------------------------------- transform


def test_to_limbs_shape_and_roundtrip(recipe: dict) -> None:
    to_limbs = recipe["to_limbs"]
    codes = np.arange(96, dtype=np.uint8).reshape(3, 32)

    limbs = to_limbs(codes)
    assert limbs.shape == (3, 4)
    assert limbs.dtype == np.int64

    # Big-endian: the first byte is the most significant of limb 0.
    expected = np.frombuffer(codes[0, :8].tobytes(), dtype=">i8")[0]
    assert limbs[0, 0] == expected

    # A 1-D query code is one row.
    assert to_limbs(codes[0]).shape == (1, 4)


def test_to_limbs_pads_short_codes(recipe: dict) -> None:
    """d=40 gives 5-byte codes; the doc claims padding is distance-neutral."""
    to_limbs = recipe["to_limbs"]
    a = np.array([[0b1111_0000, 0, 0, 0, 0]], dtype=np.uint8)
    b = np.array([[0b1111_0001, 0, 0, 0, 0]], dtype=np.uint8)
    assert to_limbs(a).shape == (1, 1)
    x = np.bitwise_xor(to_limbs(a)[0, 0], to_limbs(b)[0, 0]).view(np.uint64)
    assert bin(int(x)).count("1") == 1, "zero padding must not add set bits"


def test_to_limbs_covers_stacked_codes(recipe: dict) -> None:
    sq = StackedSignBitQuantizer(d=128, k=3, seed=1)
    codes = sq.encode(np.random.default_rng(0).standard_normal((10, 128)))
    assert codes.shape == (10, 48)
    assert recipe["to_limbs"](codes).shape == (10, 6)


# ---------------------------------------------------------- generated SQL


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    """A small centered corpus plus one encoded query, shared by the SQL tests."""
    seed, n, d = 42, 400, 256
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((n, d)).astype(np.float32)
    ids = [f"S2:{100000 + i}" for i in range(n)]

    path = tmp_path_factory.mktemp("athena") / "idx"
    corpus = Corpus.build(str(path), vectors, ids, seed=seed, center=True)

    query = rng.standard_normal(d).astype(np.float32)
    q = SignBitQuantizer(d=corpus.d, seed=seed, rotation=corpus.rotation)
    code = q.encode(query - corpus.mean)
    yield corpus, query, code
    corpus.close()


def test_generated_sql_shape(recipe: dict, built) -> None:
    corpus, _, code = built
    sql = recipe["hamming_sql"](code, "specter2_codes", k=25)
    assert sql.count("bit_count(") == corpus.d // 64 == 4
    assert "FROM   specter2_codes" in sql
    assert "ORDER BY hamming_dist" in sql
    assert sql.rstrip().endswith("LIMIT  25")


def test_generated_sql_reproduces_corpus_search(recipe: dict, built) -> None:
    """The whole point: Athena's ranking must be remax's ranking."""
    corpus, query, code = built
    k = 25

    sql = recipe["hamming_sql"](code, "specter2_codes", k=k)
    limbs = recipe["to_limbs"](corpus.codes)
    distances = _trino_distances(limbs, sql)

    # ORDER BY hamming_dist, pos — the tie-break the recipe documents.
    order = np.lexsort((np.arange(len(distances)), distances))[:k]
    athena = [(int(pos), int(distances[pos])) for pos in order]

    local = corpus.search(query, k=k)
    expected = [(corpus.lookup(r.record_id), r.distance) for r in local]

    assert athena == expected


def test_a_wrong_transform_would_be_caught(recipe: dict, built) -> None:
    """Guard the guard.

    The test above is only meaningful if a broken limb transform actually
    fails it. Byte-swapping the corpus side (the exact mistake a reader makes
    by reaching for ``<i8`` instead of ``>i8``) must change the ranking —
    otherwise this file would pass no matter what the doc said.
    """
    corpus, _, code = built
    sql = recipe["hamming_sql"](code, "t", k=10)

    good = _trino_distances(recipe["to_limbs"](corpus.codes), sql)
    swapped = recipe["to_limbs"](corpus.codes).byteswap()
    bad = _trino_distances(swapped, sql)

    assert not np.array_equal(good, bad), (
        "a byte-swapped corpus produced identical distances; the SQL "
        "evaluation in this test is not actually sensitive to the transform"
    )
