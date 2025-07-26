from sympy import Basic, Function, Add, Mul, Sum, Pow, Symbol, IndexedBase, Indexed, Number, Order

def is_random_expr(expr: Basic) -> bool:
    """
    Recursively determine if an expression involves randomness.
    """
    if isinstance(expr, ExpVal):
        return False  # 𝔼[X] is deterministic, even if X is random
    
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

class LinearExpectationMixin:
    @staticmethod
    def linear_eval(expr, Ecls, *params):

        # Flatten E(E(...))
        if isinstance(expr, Ecls):
            return Ecls(expr.args[0], *params)

        # Linearity over Add
        if isinstance(expr, Add):
            return Add(*[Ecls(arg, *params) for arg in expr.args])

        # Linearity over Sum
        if isinstance(expr, Sum):
            return Sum(Ecls(expr.function, *params), *expr.limits)

        # Product: split random and deterministic
        if isinstance(expr, Mul):
            rand_terms, det_terms = extract_random_and_deterministic(expr)

            if not rand_terms:
                return expr  # All deterministic
            elif not det_terms:
                return  # All random: unevaluated

            rand_expr = Mul(*rand_terms)
            det_expr = Mul(*det_terms)

            return det_expr * Ecls(rand_expr, *params)
        
        # Atomic non-random cases: pass through unchanged
        if isinstance(expr, (Symbol, IndexedBase, Indexed, Number, Order)):
            if not getattr(expr, 'is_random', False):
                return expr

        return  # default or Atomic random variable: unevaluated

class ExpVal(Function, LinearExpectationMixin):
    """
    Abstract symbolic expectation operator E[⋅].

    This base class implements:
    - Linearity: E[a + b] = E[a] + E[b]
    - Constant factorization: E[c⋅X] = c⋅E[X] if c is deterministic
    - Default: leaves expectation unevaluated for random variables
    """
    def _latex(self, printer):
        return r"\mathbb{E}\left[" + printer._print(self.args[0]) + r"\right]"
    
    def _sympystr(self, printer):
        return r"𝔼[" + printer._print(self.args[0]) + r']'

    @classmethod
    def eval(cls, expr):
        return cls.linear_eval(expr, cls)
    
    def _eval_derivative(self, sym):
        # Forward derivative inside
        f = self.args[0]
        new_args = (f.diff(sym), *self.args[1:])  # Replace only the integrand
        return self.func(*new_args)


