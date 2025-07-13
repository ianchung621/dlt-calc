from .base import TensorIndexedBase, TensorIndexed
from .symmetry import SymmetryMixin, SymmetryIndexedBase
from .canonical import canonicalize_dummy_indices

TensorIndexed.canonicalize = SymmetryMixin.canonicalize