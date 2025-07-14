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
    if not seq:
        return []

    partitions: list[list[list[sp.Symbol]]] = [[[seq[0]]]]

    for symbol in seq[1:]:
        new_partitions = []
        for part in partitions:
            # Insert symbol into each subset
            for i in range(len(part)):
                new_part = [subset.copy() for subset in part]
                new_part[i].append(symbol)
                new_partitions.append(new_part)
            # Add symbol as its own subset
            new_partitions.append(part + [[symbol]])
        partitions = new_partitions

    return [p for p in partitions if len(p) > 1]

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