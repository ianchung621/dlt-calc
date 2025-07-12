from sympy import Symbol, IndexedBase, Indexed

class RandomSymbol(Symbol):
    def __new__(cls, name, **kwargs):
        obj = Symbol.__new__(cls, name, **kwargs)
        obj.is_random = True
        return obj

class RandomIndexedBase(IndexedBase):
    def __new__(cls, label, **kwargs):
        obj = IndexedBase.__new__(cls, label, **kwargs)
        obj.is_random = True
        return obj
    
    def __getitem__(self, indices):
        # Ensure tuple form for multiple indices
        if not isinstance(indices, tuple):
            indices = (indices,)
        return RandomIndexed(self, *indices)

class RandomIndexed(Indexed):
    def __new__(cls, base, *args):
        obj = Indexed.__new__(cls, base, *args)
        obj.is_random = True
        return obj
