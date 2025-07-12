from sympy import Basic, Function, Add, Mul, Sum, Pow

def is_random_expr(expr: Basic) -> bool:
    """
    Recursively determine if an expression involves randomness.
    """
    if getattr(expr, 'is_random', False):
        return True
    if isinstance(expr, Sum):
        return is_random_expr(expr.function)

    if isinstance(expr, (Add, Mul, Pow, Function)):
        return any(is_random_expr(arg) for arg in expr.args)

    return False

def extract_random_and_deterministic(expr: Mul):
    """Split an expression into random and deterministic parts."""
    random_terms = []
    deterministic_terms = []

    for arg in expr.args:
        if is_random_expr(arg):
            random_terms.append(arg)
        else:
            deterministic_terms.append(arg)

    return random_terms, deterministic_terms


class ExpVal(Function):

    def _latex(self, printer):
        return r"\mathbb{E}\left[" + printer._print(self.args[0]) + r"\right]"
    
    def _sympystr(self, printer):
        return r"𝔼[" + printer._print(self.args[0]) + r']'

    @classmethod
    def eval(cls, expr):
        # Linearity over Add
        if isinstance(expr, Add):
            return Add(*[cls(arg) for arg in expr.args])
        
        # Linearity over Sum
        if isinstance(expr, Sum):
            inner = cls(expr.function)  # apply E to inside
            return Sum(inner, *expr.limits)
        
        # Product: split random and deterministic
        if isinstance(expr, Mul):
            rand_terms, det_terms = extract_random_and_deterministic(expr)

            if not rand_terms:
                return expr  # All deterministic
            elif not det_terms:
                return  # All random: unevaluated

            rand_expr = Mul(*rand_terms)
            det_expr = Mul(*det_terms)

            return det_expr * cls(rand_expr)

        # Fully deterministic
        #if not getattr(expr, 'is_random', False):
            #return expr

        return  # Atomic random variable: unevaluated


