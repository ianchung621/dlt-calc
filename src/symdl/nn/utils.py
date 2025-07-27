from sympy import symbols

from .indexed import SampleIdx, NeuronIdx
from ..tensor.indexed import TensorIdx

def _tensor_symbols(expr: str, cls) -> list[TensorIdx]|TensorIdx:
    raw_symbols = symbols(expr, seq=True)
    indices = [cls(str(s)) for s in raw_symbols]
    return indices[0] if len(indices) == 1 else indices


def sample_indices(expr: str) -> list[SampleIdx]:
    """Generate one or more SampleIdx from string expression."""
    return _tensor_symbols(expr, SampleIdx)


def neuron_indices(expr: str) -> list[NeuronIdx]:
    """Generate one or more NeuronIdx from string expression."""
    return _tensor_symbols(expr, NeuronIdx)