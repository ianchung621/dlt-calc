from sympy import Symbol, IndexedBase, Indexed

class GaussianSymbol(Symbol):
    def __new__(cls, name, **kwargs):
        obj = Symbol.__new__(cls, name, **kwargs)
        obj.is_random = True
        obj.is_gaussian = True
        return obj

class GaussianIndexedBase(IndexedBase):
    def __new__(cls, label, **kwargs):
        obj = IndexedBase.__new__(cls, label, **kwargs)
        obj.is_random = True
        obj.is_gaussian = True
        return obj

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return GaussianIndexed(self, *indices)

class GaussianIndexed(Indexed):
    def __new__(cls, base, *args):
        obj = Indexed.__new__(cls, base, *args)
        obj.is_random = True
        obj.is_gaussian = True
        return obj
