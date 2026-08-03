"""remax — rank-correct cosine LSH with a stacked-precision ladder.

Public surface (v0.1.0 in progress):

* :class:`SignBitQuantizer` — 1-bit Charikar/SimHash core.
* :class:`StackedSignBitQuantizer` — k-stack precision ladder
  (variance ∝ 1/k, every step rank-correct).
* Functional primitives: :func:`haar_rotation`, :func:`rht_rotation`,
  :func:`encode_signs`, :func:`hamming_distances`.
* :func:`characterize` — encoder characterization utility; sweeps a
  strategy × k grid and reports the recommended operating point.

remax targets *dense* embedding compression. A sparse-input
sign-packed count-sketch path was explored and removed — see
``bench/results/BM25_SKETCH.md`` for the negative result.
"""

from .characterize import CharacterizeReport, characterize
from .core import (
    SignBitQuantizer,
    asymmetric_scores,
    encode_signs,
    haar_rotation,
    hamming_distances,
    rht_rotation,
)
from .corpus import Corpus, Result
from .stacked import StackedSignBitQuantizer
from ._native import AVAILABLE as NATIVE_AVAILABLE

try:  # pragma: no cover - trivial dispatch
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("remax")
except PackageNotFoundError:  # bare source checkout, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "SignBitQuantizer",
    "StackedSignBitQuantizer",
    "asymmetric_scores",
    "Corpus",
    "Result",
    "haar_rotation",
    "rht_rotation",
    "encode_signs",
    "hamming_distances",
    "characterize",
    "CharacterizeReport",
    "NATIVE_AVAILABLE",
    "__version__",
]
