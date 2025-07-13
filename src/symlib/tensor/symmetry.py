from itertools import permutations, product
from typing import Literal

from .base import TensorIndexed, TensorIndexedBase

from itertools import permutations, product


class SymmetryMixin:
    def canonicalize(self: TensorIndexed):
        indices = list(self.indices)
        symmetries = getattr(self, 'symmetries', None)

        if symmetries is None:
            return self

        if symmetries == "full":
            candidates = [
                type(self)(self.base, *perm)
                for perm in permutations(indices)
            ]
            return min(candidates, key=lambda x: x.sort_key())

        if isinstance(symmetries, list):
            swap_rules: list[dict] = []
            for group in symmetries:
                # group can be either:
                #   (0, 1)                → index symmetry, i.e., i ↔ j
                #   ([0, 1], [2, 3])      → pairwise symmetry, i.e., (i,j) ↔ (k,l)

                if isinstance(group[0], int):
                    # Index symmetry: (0, 1)
                    i, j = group
                    a, b = indices[i], indices[j]
                    swap_rules.append({a: b, b: a})

                elif isinstance(group[0], list):
                    # Pairwise symmetry: ([0, 1], [2, 3])
                    g1, g2 = group
                    assert len(g1) == len(g2), "Pairwise symmetry groups must have same length: ([i,j], [k,l])"
                    a = [indices[i] for i in g1]
                    b = [indices[i] for i in g2]
                    rule = dict(zip(a, b)) | dict(zip(b, a))  # bidirectional mapping
                    swap_rules.append(rule)

                else:
                    raise TypeError("Unsupported symmetry format")

            def apply_swap_mapping(base_indices, active_swaps: list[bool]):
                result = list(base_indices)
                for swap_active, rule in zip(active_swaps, swap_rules):
                    if swap_active:
                        result = [rule.get(idx, idx) for idx in result]
                return result

            best = None
            for bits in product([False, True], repeat=len(swap_rules)):
                candidate_indices = apply_swap_mapping(indices, bits)
                candidate = type(self)(self.base, *candidate_indices)
                if best is None or candidate.sort_key() < best.sort_key():
                    best = candidate

            return best

        return self

class SymmetryIndexedBase(TensorIndexedBase, SymmetryMixin):
    def __new__(
        cls,
        label,
        *,
        symmetries: list[tuple[int, int] | tuple[list[int], list[int]]] | Literal["full"] | None = None,
        **kwargs
    ):
        return super().__new__(cls, label, symmetries=symmetries, **kwargs)
