import sympy as sp
from typing import Iterable, Callable, Optional, Tuple

def wilds(names: str,
          exclude: Optional[Iterable] = None,
          properties: Optional[Iterable[Callable]] = None) -> Tuple[sp.Wild, ...]:
    """
    Create multiple Wild symbols given a space/comma-separated string.

    Parameters:
        names (str): Symbol names, e.g., "x y z"
        exclude (iterable, optional): Symbols or expressions to exclude from matching
        properties (iterable of callables, optional): Each callable takes an expression and returns bool

    Returns:
        tuple of sympy.Wild objects
    """
    exclude = tuple(sp.sympify(x) for x in exclude) if exclude is not None else ()
    properties = tuple(properties) if properties is not None else ()

    return tuple(
        sp.Wild(name.strip(), exclude=exclude, properties=properties)
        for name in names.replace(',', ' ').split()
    )

def wild_subs(expr: sp.Basic, rule_dict: dict):
    """
    Apply a dictionary of pattern-based Wild substitutions, like .subs() but using .replace().
    
    Parameters
    ----------
    expr : sympy.Basic
        The symbolic expression to rewrite
    rule_dict : dict
        Dictionary of {pattern: replacement}, where pattern can contain Wild symbols

    Returns
    -------
    sympy.Basic
        Expression with all matching patterns replaced
    """
    for pattern, repl in rule_dict.items():
        expr = expr.replace(pattern, repl)
    return expr


