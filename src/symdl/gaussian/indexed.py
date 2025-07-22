from sympy import Symbol
from ..tensor import TensorIndexedBase

class GaussianIndexedBase(TensorIndexedBase):
    def __new__(cls, label, **kwargs):
        return super().__new__(cls, label, is_random=True, is_gaussian=True, **kwargs)

class GaussianSymbol(Symbol):
    def __new__(cls, name, **kwargs):
        obj = Symbol.__new__(cls, name, **kwargs)
        obj.is_random = True
        obj.is_gaussian = True
        return obj
