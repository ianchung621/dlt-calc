from sympy import IndexedBase, Indexed, Symbol
from typing import Self

from ..assumption import AssumptionMixin

class TensorIndexedBase(IndexedBase, AssumptionMixin):
    def __new__(cls, label, *, is_random=False, is_gaussian=False, symmetries=None, **kwargs):
        obj = super().__new__(cls, label, **kwargs)

        obj.symmetries = symmetries
        facts = {
            "is_random": is_random,
            "is_gaussian": is_gaussian,
        }
        for key, value in obj._inject_facts(facts).items():
            setattr(obj, key, value)

        return obj

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorIndexed(self, *indices)

class TensorIndexed(Indexed, AssumptionMixin):
    def __new__(cls, base, *indices):
        obj = super().__new__(cls, base, *indices)
        obj.symmetries = getattr(base, "symmetries", None)
        obj.apply_assumptions(base)
        return obj
    
    def canonicalize(self) -> Self: # implemented in .symmetry.py
        raise NotImplementedError("You must inject or override canonicalize()")

class TensorIdx(Symbol):

    is_integer = True
    is_real = True
    is_finite = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True
    
    def __new__(cls, name: str, *, is_up: bool = False, **kwargs):
        obj = super().__new__(cls, name, **kwargs)
        obj.is_up = is_up
        return obj
