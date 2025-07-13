from sympy import IndexedBase, Indexed
from typing import Self

class TensorIndexedBase(IndexedBase):
    def __new__(cls, label, *, is_random=False, is_gaussian=False, symmetries=None, **kwargs):
        obj = super().__new__(cls, label, **kwargs)
        obj.is_random = is_random
        obj.is_gaussian = is_gaussian
        obj.symmetries = symmetries
        return obj

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorIndexed(self, *indices)

class TensorIndexed(Indexed):
    def __new__(cls, base, *indices):
        obj = super().__new__(cls, base, *indices)
        obj.is_random = getattr(base, "is_random", False)
        obj.is_gaussian = getattr(base, "is_gaussian", False)
        obj.symmetries = getattr(base, "symmetries", None)
        return obj
    
    def canonicalize(self) -> Self: # implemented in .symmetry.py
        raise NotImplementedError("You must inject or override canonicalize()")
