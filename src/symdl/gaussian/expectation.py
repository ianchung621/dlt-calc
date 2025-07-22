import sympy as sp

from ..random import ExpVal
from ..random.expectation import LinearExpectationMixin

class GaussianEval(ExpVal):
    nargs = (2,)  # (expr, K)

    @classmethod
    def eval(cls, expr, K):
        return cls.linear_eval(expr, cls, K)

    def _latex(self, printer):
        expr, K = self.args
        return r"\left\langle " + printer._print(expr) + r"\right\rangle_{" + printer._print(K) + r"}"
    
    def _sympystr(self, printer):
        expr, K = self.args
        return f"⟨{printer.doprint(expr)}⟩_{printer.doprint(K)}"

class GaussianExpVal:
    """
    Callable expectation operator ⟨⋅⟩_K for Gaussian distributions.

    Parameters
    ----------
    K : sp.Symbol or sp.IndexBase
        The kernel (inverse covariance) matrix for the Gaussian.

    Returns
    -------
    GaussianEval: sp.Expr
        A symbolic expectation operator that evaluates ⟨expr⟩_K.

    Examples
    --------
    >>> from symlib.random import RandomIndexedBase
    >>> K = sp.IndexedBase('K')
    >>> z = RandomIndexedBase('z')
    >>> mu, nu = sp.symbols('mu nu', integer=True)
    >>> EK = GaussianExpVal(K)
    >>> EK(z[mu] * z[nu])
    ⟨ z_μ z_ν ⟩_K
    """
    def __init__(self, K: sp.Symbol|sp.IndexedBase):
        self.K = K

    def __call__(self, expr) -> sp.Expr:
        return GaussianEval(expr, self.K)

