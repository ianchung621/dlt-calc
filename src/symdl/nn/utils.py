from sympy import symbols

from .indexed import SampleIdx, NeuronIdx
from ..tensor.indexed import TensorIdx

def _tensor_symbols(expr: str, cls, *, is_up: bool = False) -> list[TensorIdx]|TensorIdx:
    raw_symbols = symbols(expr, seq=True)
    indices = [cls(str(s), is_up=is_up) for s in raw_symbols]
    return indices[0] if len(indices) == 1 else indices


def sample_indices(expr: str, *, is_up: bool = False) -> list[SampleIdx]:
    """Generate one or more SampleIdx from string expression."""
    return _tensor_symbols(expr, SampleIdx, is_up=is_up)


def neuron_indices(expr: str, *, is_up: bool = False) -> list[NeuronIdx]:
    """Generate one or more NeuronIdx from string expression."""
    return _tensor_symbols(expr, NeuronIdx, is_up=is_up)