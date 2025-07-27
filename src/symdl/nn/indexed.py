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

        upper_str = format_group(self.up_indices)
        lower_str = format_group(self.down_indices)

        base_str = printer._print(self.base)

        return self.format_latex(base_str, upper_str, lower_str)

class NNIndexedBase(TensorIndexedBase):
    def __getitem__(self, indices) -> NNIndexed:
        if not isinstance(indices, tuple):
            indices = (indices,)
        return NNIndexed(self, *indices)
