from sympy import Symbol
from ..tensor import TensorIndexedBase

class RandomIndexedBase(TensorIndexedBase):
    def __new__(cls, label, **kwargs):
        return super().__new__(cls, label, is_random=True, **kwargs)

class RandomSymbol(Symbol):
    def __new__(cls, name, **kwargs):
        obj = Symbol.__new__(cls, name, **kwargs)
        obj.is_random = True
        return obj