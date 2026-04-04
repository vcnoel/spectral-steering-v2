import torch
import torch.nn as nn
from typing import List, Dict

class RepEHook:
    """
    PyTorch forward hook for Representation Engineering (Zou et al., 2023).
    Adds/subtracts a reading vector at specified layers during inference.
    """
    def __init__(self, reading_vectors: Dict[int, torch.Tensor], alpha: float = 1.0):
        self.reading_vectors = reading_vectors  # Mapping from layer_idx to vector
        self.alpha = alpha
        self.handles = []

    def _get_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # output is usually (hidden_states, optional_tuple...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            direction = self.reading_vectors[layer_idx].to(hidden_states.device, dtype=hidden_states.dtype)
            
            # RepE intervention: Add alpha * direction
            hidden_states = hidden_states + self.alpha * direction
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        return hook

    def register(self, model: nn.Module, layer_names: List[str]):
        """
        Registers the hook on the specified layers.
        Usually targets the residual stream (e.g., standard transformer blocks).
        """
        for idx, (name, module) in enumerate(model.named_modules()):
            if name in layer_names and idx in self.reading_vectors:
                handle = module.register_forward_hook(self._get_hook(idx))
                self.handles.append(handle)

    def remove(self):
        """Removes all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
