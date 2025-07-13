from typing import Iterator
from itertools import permutations, product
from collections import defaultdict

from sympy import Expr, Symbol, Sum
from .base import TensorIndexed

def canonicalize_expr(expr: Expr) -> Expr:
    """Canonicalize all TensorIndexed terms in a Expr"""
    return expr.replace(
        lambda x: isinstance(x, TensorIndexed),
        lambda x: x.canonicalize()
    )

def permute_limit_order(expr: Sum, new_order: list[Symbol]) -> Sum:
    """
    Given a Sum(expr, (i1, dom1), (i2, dom2), ...), return an equivalent Sum
    with dummy indices permuted to match `new_order`.

    Parameters
    ----------
    expr : Sum
        A sympy Sum expression over dummy indices.
    new_order : list[Symbol]
        A permutation of the original dummy index symbols (same symbols, different order).

    Returns
    -------
    Sum
        A new Sum object with permuted index symbols, the order of limits will be sorted

    Example
    -------
    expr = Sum(f(i, j), (i, 0, N), (j, 0, N))
    new = permute_limit_order(expr, [j, i])  # Sum(f(j, i), (j, 0, N), (i, 0, N))
    """
    assert isinstance(expr, Sum)
    old_limits = expr.limits
    old_syms = [lim[0] for lim in old_limits]
    old_domains = [lim[1:] for lim in old_limits]
    assert len(old_limits) == len(new_order), "new_order must match number of limits"
    assert set(old_syms) == set(new_order), f"Mismatch between original symbols {old_syms} and new_order {new_order}"

    rename_map = {old: new for old, new in zip(old_syms, new_order)}

    for old, new in zip(old_syms, new_order):
        if old == new:
            continue
        old_domain = old_limits[old_syms.index(old)][1:]
        new_domain = old_limits[old_syms.index(new)][1:]
        assert old_domain == new_domain, f"Domain mismatch for {old} → {new}: {old_domain} != {new_domain}"
        
    new_func = expr.function.xreplace(rename_map)

    renamed_limits = [(rename_map[old], *dom) for old, dom in zip(old_syms, old_domains)]
    sorted_limits = sorted(renamed_limits, key=lambda lim: lim[0].name)

    return Sum(new_func, *sorted_limits)

def extract_group_indices(limits: list[tuple]) -> list[int]:
    """
    Assign each limit a group index based on domain equality.

    Parameters
    ----------
    limits : list of (symbol, *domain)

    Returns
    -------
    list[int]
        Group index assigned to each limit
    """
    domain_to_gid = {}
    group_ids = []
    next_gid = 0

    for _, *domain in limits:
        dom = tuple(domain)
        if dom not in domain_to_gid:
            domain_to_gid[dom] = next_gid
            next_gid += 1
        group_ids.append(domain_to_gid[dom])
    return group_ids

def valid_permutations_by_group(symbols: list[Symbol], group_ids: list[int]) -> list[list[Symbol]]:
    """
    Generate all symbol permutations that preserve group_id order.

    Parameters
    ----------
    symbols : list[Symbol]
    group_ids : list[int]

    Returns
    -------
    list[list[Symbol]]
        Permuted symbols lists where group structure is preserved.
    """
    assert len(symbols) == len(group_ids)

    group_to_symbols = defaultdict(list) # gid: symbols
    for sym, gid in zip(symbols, group_ids):
        group_to_symbols[gid].append(sym)
    
    def assign_group_values(group_ids: list[int], gid_values: dict[int, list]) -> list:
        # group_ids = [0,1,1,0, ..] gid_values = {0: [a,b,..], 1: [x,y,..]}
        # -> group_values = [a, x, y, b, ..]
        gid_iters: dict[int, Iterator] = {gid: iter(vals) for gid, vals in gid_values.items()}
        group_values = [next(gid_iters[gid]) for gid in group_ids]
        return group_values 

    sorted_gids = sorted(group_to_symbols)
    group_perms = [list(permutations(group_to_symbols[gid])) for gid in sorted_gids]

    results = []
    for perm_combo in product(*group_perms):
        gid_values = {gid: list(perm) for gid, perm in zip(sorted_gids, perm_combo)}
        results.append(assign_group_values(group_ids, gid_values))

    return results

def get_dummy_index_options(expr: Sum) -> list[Sum]:
    limits = list(expr.limits)
    symbols = [lim[0] for lim in limits]
    group_ids = extract_group_indices(limits)
    valid_orders = valid_permutations_by_group(symbols, group_ids)
    return [permute_limit_order(expr, new_order) for new_order in valid_orders]

def canonicalize_dummy_indices(expr: Expr) -> Expr:
    """
    Recursively canonicalize all Sum(...) nodes in an expression by:
    - Enumerating all dummy index permutations that preserve domain grouping
    - Canonicalizing each variant (tensor-wise)
    - Choosing the minimal one by SymPy's sort_key()

    Parameters
    ----------
    expr : Expr
        A sympy expression potentially containing Sum(...) nodes.

    Returns
    -------
    Expr
        Expression with all Sum(...) nodes canonicalized under dummy index symmetry.
    """
    return expr.replace(
        lambda e: isinstance(e, Sum),
        lambda e: min(
            (canonicalize_expr(opt) for opt in get_dummy_index_options(e)),
            key=lambda opt: opt.sort_key()
        )
    )
