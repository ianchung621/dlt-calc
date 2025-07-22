from .base import TensorIndexedBase, TensorIndexed, TensorIdx
from .symmetry import SymmetryMixin, SymmetryIndexedBase
from .canonical import canonicalize_dummy_indices

TensorIndexed.canonicalize = SymmetryMixin.canonicalize