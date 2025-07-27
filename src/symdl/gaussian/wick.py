from typing import Iterable

import sympy as sp
from ..random import ExpVal

def wick(expr: ExpVal):
    """
    Apply Wick's theorem to ExpVal of a product of random variables.
    Args:
        expr: an ExpVal(Mul(...)) with all terms being RVs

    Returns:
        If any RV is non-Gaussian, leave it
        
        else, return sum of pairwise contractions (ExpVal(a*b) * ExpVal(c*d) + ...) or 0 if odd order
    """
    if not isinstance(expr, ExpVal):
        raise TypeError("wick() expects an ExpVal(...)")

    inner = expr.args[0] # RVs 

    if not isinstance(inner, sp.Mul): # parse pruduct rv1 * rv2 * ... only
        return expr

    terms = flatten_rvs(inner.args) # [rv1, rv2, ...]

    if not all(getattr(t, 'is_gaussian', False) for t in terms): # make sure all RVs are Gaussian
        return expr

    if len(terms) % 2 != 0:
        return 0  # odd moments vanish

    def pairings(xs):
        if not xs:
            yield []
        else:
            x0 = xs[0]
            for i in range(1, len(xs)):
                for rest in pairings(xs[1:i] + xs[i+1:]):
                    yield [(x0, xs[i])] + rest

    # Determine expectation class (ExpVal, GaussianEval, etc.)
    Ecls = type(expr)

    return sp.Add(*[
        sp.Mul(*[Ecls(a * b, *expr.args[1:]) for a, b in pair])
        for pair in pairings(terms)
    ])

def flatten_rvs(args: Iterable) -> list:

    def is_valid_exponent(exp) -> bool:
        if isinstance(exp, (int, sp.Integer)):
            return exp >= 0
        if isinstance(exp, sp.Symbol):
            return exp.is_integer and (exp.is_nonnegative is not False)
        return False

    flat = []
    for term in args:
        if isinstance(term, sp.Pow):
            base, exp = term.args
            if not is_valid_exponent(exp):
                raise ValueError("Only non-negative integer powers are supported in Wick expansion.")
            flat.extend([base] * exp)
        else:
            flat.append(term)
    return flat

def wick_contraction(expr: sp.Expr) -> sp.Expr:
    """
    Recursively apply Wick contraction to all ExpVal(...) nodes in the expression.
    """
    return expr.replace(
        lambda e: isinstance(e, ExpVal),
        lambda e: wick(e)
    )