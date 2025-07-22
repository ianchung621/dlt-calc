import sympy as sp
from .indexed import NNIndexedBase

class ZLayer:
    _counter = 0  

    def __init__(self, 
                 W: sp.IndexedBase | NNIndexedBase, 
                 b: sp.IndexedBase | NNIndexedBase,
                 x_in: sp.IndexedBase | NNIndexedBase, 
                 n_in: sp.Symbol | int):
        self.W = W
        self.b = b
        self.x_in = x_in
        self.n_in = n_in
        self._C_b = None
        self._C_w = None

    def gaussian_init(self, C_w: sp.Symbol|float, C_b: sp.Symbol|float):
        self._C_w = C_w
        self._C_b = C_b

    def _get_dummy_index(self, prefix='k'):
        """Generate a unique dummy index each call."""
        ZLayer._counter += 1
        return sp.Symbol(f'{prefix}_{ZLayer._counter}', integer=True)

    def __getitem__(self, idx):
        i, alpha = idx
        k = self._get_dummy_index()
        return self.b[i] + sp.Sum(self.W[i, k] * self.x_in[k, alpha], (k, 1, self.n_in))
    
    @property
    def layer_metric(self):
        if self._C_b is None or self._C_w is None:
            raise ValueError("Call gaussian_init(C_w, C_b) first before using .layer_metric")

        return _MetricProxy(self)

class _MetricProxy:

    def __init__(self, outer: ZLayer):
        self.outer = outer

    def __getitem__(self, idx_pair):
        alpha, beta = idx_pair
        k = self.outer._get_dummy_index()
        return (
            self.outer._C_b
            + (self.outer._C_w / self.outer.n_in)
            * sp.Sum(
                self.outer.x_in[k, alpha] * self.outer.x_in[k, beta],
                (k, 1, self.outer.n_in)
            )
        )