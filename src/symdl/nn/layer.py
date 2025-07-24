import sympy as sp
from .indexed import NNIndexedBase, NeuronIdx

class Layer:
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
        Layer._counter += 1
        return NeuronIdx(f'{prefix}_{Layer._counter}')

    @property
    def preactivation(self):
        return _PreactivationProxy(self)
    
    @property
    def layer_metric(self):
        if self._C_b is None or self._C_w is None:
            raise ValueError("Call gaussian_init(C_w, C_b) first before using .layer_metric")

        return _MetricProxy(self)

class _PreactivationProxy:

    def __init__(self, outer: Layer):
        self.outer = outer
    
    def __getitem__(self, idx_pair):
        i, alpha = idx_pair
        k = self.outer._get_dummy_index()
        return (self.outer.b[i]
                + sp.Sum(
                    self.outer.W[i, k] * self.outer.x_in[k, alpha],
                    (k, 1, self.outer.n_in)
                    )
                )

class _MetricProxy:

    def __init__(self, outer: Layer):
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