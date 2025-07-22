import sympy as sp

def pull_sums_out_front(expr: sp.Basic) -> sp.Basic:
    """
    Recursively pulls all Sum(...) objects out front as a single multi-indexed Sum.
    Assumes independence between summation indices.
    """
    if isinstance(expr, sp.Sum):
        # Already a Sum — flatten inner expression
        inner = pull_sums_out_front(expr.function)
        return sp.Sum(inner, *expr.limits)
    
    if expr.is_Atom:
        return expr

    if expr.func in {sp.Mul, sp.Add}:
        
        new_args = [pull_sums_out_front(arg) for arg in expr.args]

        # For multiplication: extract Sum factors
        if expr.func == sp.Mul:
            sum_factors = [arg for arg in new_args if isinstance(arg, sp.Sum)]
            non_sum_factors = [arg for arg in new_args if not isinstance(arg, sp.Sum)]
            
            if not sum_factors:
                return sp.Mul(*new_args)
            
            # Combine all Sum limits and product of Sum functions
            combined_limits = []
            combined_func = 1
            for s in sum_factors:
                combined_limits.extend(s.limits)
                combined_func *= s.function

            final_expr = sp.Mul(*non_sum_factors) * combined_func
            return sp.Sum(final_expr, *combined_limits)
        
        return expr.func(*new_args)

    # Catch-all: recurse into function arguments
    return expr.func(*[pull_sums_out_front(arg) for arg in expr.args])

def pull_coef_out_sum(expr: sp.Basic) -> sp.Basic:
    """
    Recursively pulls constant coefficients out of Sum(...) expressions.
    Assumes independence between coefficients and summation indices.
    """
    if isinstance(expr, sp.Sum):
        inner = pull_coef_out_sum(expr.function)

        # Identify parts of the function that are independent of summation indices
        free_vars = set().union(*(set(lim[0].free_symbols) for lim in expr.limits))
        if inner.is_Mul:
            coeffs = []
            summand_parts = []
            for factor in inner.args:
                if factor.free_symbols.isdisjoint(free_vars):
                    coeffs.append(factor)
                else:
                    summand_parts.append(factor)
            if coeffs:
                pulled = sp.Mul(*coeffs)
                remaining = sp.Mul(*summand_parts)
                return pulled * sp.Sum(remaining, *expr.limits)
        return sp.Sum(inner, *expr.limits)

    if expr.is_Atom:
        return expr

    # Apply recursively to args
    return expr.func(*[pull_coef_out_sum(arg) for arg in expr.args])


def sum_kronecker_contract(expr: sp.Basic) -> sp.Basic:
    """
    Simplifies expressions by contracting KroneckerDelta(i, j) within sums.
    Example: Sum(Sum(δ_ij f(i, j), (i, ...)), (j, ...)) → Sum(f(i, i), (i, ...))
    """

    if isinstance(expr, sp.Sum):
        # Recursively contract inside sum body
        new_body = sum_kronecker_contract(expr.function)
        new_limits = list(expr.limits)

        # Try to find KroneckerDelta(i, j) where both i and j are summation indices
        for delta in new_body.atoms(sp.KroneckerDelta):
            i, j = delta.args
            sum_indices = [lim[0] for lim in new_limits]
            if i in sum_indices and j in sum_indices:
                # Replace j by i in body, remove delta, drop redundant limit
                reduced_body = new_body.subs(j, i).replace(delta, 1)

                # Remove one of the duplicated limits
                new_limits = [lim for lim in new_limits if lim[0] != j]

                return sum_kronecker_contract(sp.Sum(reduced_body, *new_limits))

        # Return reconstructed Sum with possibly simplified body
        return sp.Sum(new_body, *new_limits)

    elif isinstance(expr, sp.Add):
        return sp.Add(*[sum_kronecker_contract(arg) for arg in expr.args])

    elif isinstance(expr, sp.Mul):
        return sp.Mul(*[sum_kronecker_contract(arg) for arg in expr.args])

    else:
        return expr