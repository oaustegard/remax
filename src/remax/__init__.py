"""remax — rank-correct cosine LSH with a stacked-precision ladder.

Public surface (v0.1.0 in progress):

* :class:`SignBitQuantizer` — 1-bit Charikar/SimHash core (this issue).
* Functional primitives: :func:`haar_rotation`, :func:`encode_signs`,
  :func:`hamming_distances`, :func:`hamming_search`.

Multi-precision stacking (``StackedSignBitQuantizer``) is tracked in
issue #3 and not yet exported.
"""

from .core import (
    SignBitQuantizer,
    encode_signs,
    haar_rotation,
    hamming_distances,
    hamming_search,
)

__version__ = "0.0.0"

__all__ = [
    "SignBitQuantizer",
    "haar_rotation",
    "encode_signs",
    "hamming_distances",
    "hamming_search",
    "__version__",
]
