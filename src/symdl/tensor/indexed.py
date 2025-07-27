from sympy import IndexedBase, Indexed, Symbol
from typing import Self, Literal

from ..assumption import AssumptionMixin

class TensorIndexedBase(IndexedBase, AssumptionMixin):
    def __new__(
        cls, 
        label,
        *,
        is_random=False,
        is_gaussian=False,
        symmetries: list[tuple[int, int] | tuple[list[int], list[int]]] | Literal["full"] | None = None,
        idx_is_superscript: list[int] | bool = False,
        idx_types: type | list[type | None] | None = None,
        **kwargs
    ):
        """

        Parameters
        ----------
        label : str or Symbol
            The name of the tensor (e.g. 'V', 'K'). Passed to sympy.IndexedBase.
        is_random : bool, optional
            if True, Expectation Value left inevaluated 
        is_gaussian : bool, optional
            if True, assume zero mean gaussian and support Wick contraction
        symmetries : list[tuple[int, int]  |  tuple[list[int], list[int]]] | Literal[&quot;full&quot;] | None, optional
            A list of index symmetries. Supports:
            
            - **Index symmetry**: `(i, j)` implies `T[..., i, ..., j, ...] = T[..., j, ..., i, ...]`
            - **Pairwise symmetry**: `([i, j], [k, l])` implies `T[..., i, j, ..., k, l, ...] = T[..., k, l, ..., i, j, ...]`
            - **"full"**: All permutations of indices are symmetric
        idx_is_superscript : list[int] | bool, optional
            Controls how indices are rendered in LaTeX:
            
            - `False` (default): all indices are subscript
            - `True`: all indices are superscript
            - `list[int]`: list of index positions (0-based) to render as superscripts
        idx_types : type | list[type | None] | None, optional
            Expected types for each index position. Can be used to enforce typing or skip checking:

            - `type`: enforce that all indices are instances of this type
            - `list[type | None]`: per-index type checking (use `None` to skip checking for specific positions)
            - `None` (default): disable type checking entirely
        """
        obj = super().__new__(cls, label, **kwargs)

        obj.symmetries = symmetries
        obj.idx_is_superscript = idx_is_superscript
        obj.idx_types = idx_types 

        facts = {
            "is_random": is_random,
            "is_gaussian": is_gaussian,
        }
        for key, value in obj._inject_facts(facts).items():
            setattr(obj, key, value)

        return obj

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorIndexed(self, *indices)

class TensorIndexed(Indexed, AssumptionMixin):
    def __new__(cls, base, *indices):
        obj = super().__new__(cls, base, *indices)
        obj.symmetries = getattr(base, "symmetries", None)
        obj.idx_is_superscript = getattr(base, "idx_is_superscript", False)
        obj.idx_types = getattr(base, "idx_types", None)
        obj.apply_assumptions(base)

        obj._validate_index_types()

        return obj
    
    def canonicalize(self) -> Self: # implemented in .symmetry.py
        raise NotImplementedError("You must inject or override canonicalize()")
    
    @property
    def up_indices(self) -> list:
        if self.idx_is_superscript is True:
            return list(self.indices)
        elif self.idx_is_superscript is False:
            return []
        else:  # assume list[int]
            return [idx for i, idx in enumerate(self.indices) if i in self.idx_is_superscript]

    @property
    def down_indices(self) -> list:
        if self.idx_is_superscript is True:
            return []
        elif self.idx_is_superscript is False:
            return list(self.indices)
        else:  # assume list[int]
            return [idx for i, idx in enumerate(self.indices) if i not in self.idx_is_superscript]
    
    @staticmethod
    def format_latex(base_str: str, upper_str: str, lower_str: str) -> str:
        if upper_str and lower_str:
            return f"{base_str}^{{{upper_str}}}_{{{lower_str}}}"
        elif upper_str:
            return f"{base_str}^{{{upper_str}}}"
        elif lower_str:
            return f"{base_str}_{{{lower_str}}}"
        else:
            return base_str
    
    def _latex(self, printer):
        upper_str = ",".join(printer._print(i) for i in self.up_indices)
        lower_str = ",".join(printer._print(i) for i in self.down_indices)
        base_str = printer._print(self.base)
        return self.format_latex(base_str, upper_str, lower_str)


    @property
    def idx_classes(self) -> list[type]:
        if isinstance(self.idx_types, type):
            return [self.idx_types] * len(self.indices)
        if isinstance(self.idx_types, list):
            if len(self.idx_types) != len(self.indices):
                raise ValueError(
                    f"Length mismatch: idx_types has {len(self.idx_types)} types "
                    f"but there are {len(self.indices)} indices."
                )
            return self.idx_types
        return [type(i) for i in self.indices] # fallback
    
    def _validate_index_types(self):
        if self.idx_types is not None:
            for i, (idx, expected_cls) in enumerate(zip(self.indices, self.idx_classes)):
                if expected_cls is None:
                    continue
                if not isinstance(idx, expected_cls):
                    raise TypeError(
                        f"Index {i} = {idx} must be {expected_cls.__name__}, got {type(idx).__name__}"
                    )

class TensorIdx(Symbol):

    is_integer = True
    is_real = True
    is_finite = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True