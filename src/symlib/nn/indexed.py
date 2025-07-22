from ..tensor import TensorIdx, TensorIndexed, TensorIndexedBase

class SampleIdx(TensorIdx): pass
class NeuronIdx(TensorIdx): pass

class NNIndexed(TensorIndexed):
    def _latex(self, printer):
        def format_group(indices):
            neuron = [printer._print(idx) for idx in indices if isinstance(idx, NeuronIdx)]
            sample = [printer._print(idx) for idx in indices if isinstance(idx, SampleIdx)]
            others = [printer._print(idx) for idx in indices if not isinstance(idx, (NeuronIdx, SampleIdx))]
            
            parts = []
            if neuron:
                parts.append("".join(neuron))
            if sample:
                if parts:
                    parts.append(";" + "".join(sample))
                else:
                    parts.append("".join(sample))
            if others:
                if parts:
                    parts.append("," + ",".join(others))
                else:
                    parts.append(",".join(others))

            return "".join(parts)

        up_indices = [idx for idx in self.indices if getattr(idx, "is_up", False)]
        down_indices = [idx for idx in self.indices if not getattr(idx, "is_up", False)]

        upper_str = format_group(up_indices)
        lower_str = format_group(down_indices)

        base_str = printer._print(self.base)

        if upper_str and lower_str:
            return f"{base_str}^{{{upper_str}}}_{{{lower_str}}}"
        elif upper_str:
            return f"{base_str}^{{{upper_str}}}"
        elif lower_str:
            return f"{base_str}_{{{lower_str}}}"
        else:
            return base_str

class NNIndexedBase(TensorIndexedBase):
    def __getitem__(self, indices) -> NNIndexed:
        if not isinstance(indices, tuple):
            indices = (indices,)
        return NNIndexed(self, *indices)
