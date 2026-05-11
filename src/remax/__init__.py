"""remax — rank-correct cosine LSH with a stacked-precision ladder.

Public surface (v0.1.0 in progress):

* :class:`SignBitQuantizer` — 1-bit Charikar/SimHash core.
* :class:`StackedSignBitQuantizer` — k-stack precision ladder
  (variance ∝ 1/k, every step rank-correct).
* Functional primitives: :func:`haar_rotation`, :func:`encode_signs`,
  :func:`hamming_distances`, :func:`hamming_search`.
* :func:`characterize` — encoder characterization utility; sweeps a
  strategy × k grid and reports the recommended operating point.

remax targets *dense* embedding compression. A sparse-input
sign-packed count-sketch path was explored and removed — see
``bench/results/BM25_SKETCH.md`` for the negative result.
"""

from .characterize import CharacterizeReport, characterize
from .core import (
    SignBitQuantizer,
    encode_signs,
    haar_rotation,
    hamming_distances,
    hamming_search,
)
from .corpus import Corpus, Result
from .stacked import StackedSignBitQuantizer
from ._native import AVAILABLE as NATIVE_AVAILABLE

__version__ = "0.0.0"

__all__ = [
    "SignBitQuantizer",
    "StackedSignBitQuantizer",
    "Corpus",
    "Result",
    "haar_rotation",
    "encode_signs",
    "hamming_distances",
    "hamming_search",
    "characterize",
    "CharacterizeReport",
    "NATIVE_AVAILABLE",
    "__version__",
]
