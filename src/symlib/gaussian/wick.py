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
        if getattr(inner, 'is_gaussian', False):
            return 0
        return expr

    terms = list(inner.args) # [rv1, rv2, ...]

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

    return sp.Add(*[
        sp.Mul(*[ExpVal(a * b) for a, b in pair])
        for pair in pairings(terms)
    ])

def wick_contraction(expr: sp.Expr) -> sp.Expr:
    """
    Recursively apply Wick contraction to all ExpVal(...) nodes in the expression.
    """
    return expr.replace(
        lambda e: isinstance(e, ExpVal),
        lambda e: wick(e)
    )