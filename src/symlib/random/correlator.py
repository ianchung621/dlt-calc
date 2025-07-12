from typing import Generator
from functools import lru_cache, wraps

import sympy as sp

from .expectation import ExpVal

def all_nontrivial_partitions(seq: tuple[sp.Symbol, ...]) -> list[list[list[sp.Symbol]]]:
    """
    Example
    --------
    Input: 
        (a, b, c)

    Output (order not guaranteed):

        [
            [['a'], ['b', 'c']],
            [['a', 'b'], ['c']],
            [['b'], ['a', 'c']],
            [['a'], ['b'], ['c']]
        ]
    """
    def partitions(seq: tuple[sp.Symbol, ...]) -> Generator[list[list[sp.Symbol]], None, None]:
        if len(seq) == 1: 
            yield [list(seq)]
            return
        first = seq[0]
        for smaller in partitions(seq[1:]):
            for i in range(len(smaller)):
                new_subset = [first] + list(smaller[i])
                yield smaller[:i] + [new_subset] + smaller[i+1:]
            yield [[first]] + smaller
    return [p for p in partitions(seq) if len(p) > 1]

def __cached_with_doc(fn):
    return lru_cache(maxsize=None)(wraps(fn)(fn))

@__cached_with_doc
def connected_correlator(
    z: sp.IndexedBase,
    indices: tuple[sp.Symbol, ...],
    even_parity: bool = True
) -> sp.Expr: 
    """
    Recursively compute the connected correlator E[z[μ1]⋯z[μM]]_connected.

    Parameters
    ----------
    z : IndexedBase
        An indexed random variable.
    indices : tuple of sympy.Symbol
        The indices μ1, μ2, ..., μM.
    even_parity : bool
        If True, odd-order cumulants vanish.

    Returns
    -------
    Expr
        A symbolic expression for the connected correlator.
    """
    M = len(indices)

    if even_parity and M % 2 == 1:
        return 0
    
    if M == 1:
        return 0 if even_parity else ExpVal(z[indices[0]])
    if M == 2:
        i, j = indices
        return ExpVal(z[i]*z[j]) if even_parity else ExpVal(z[i]*z[j]) - ExpVal(z[i]) * ExpVal(z[j])
    
    # Start from full moment
    full: sp.Expr = ExpVal(sp.Mul(*[z[i] for i in indices]))
    # Subtract products of connected correlators over all nontrivial partitions
    for partition in all_nontrivial_partitions(indices):
        term = 1
        for subset in partition:
            term *= connected_correlator(z, tuple(subset), even_parity)
        full -= term
    return full