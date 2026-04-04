import torch
import torch.nn as nn
from typing import List

class ActAddHook:
    """
    PyTorch forward hook for Activation Addition (Turner et al., 2023).
    Adds a steering vector at a single specific layer during inference.
    """
    def __init__(self, steering_vector: torch.Tensor, target_layer_idx: int, alpha: float = 1.0):
        self.steering_vector = steering_vector
        self.target_layer_idx = target_layer_idx
        self.alpha = alpha
        self.handle = None

    def _hook(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        direction = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        
        # ActAdd intervention: Add alpha * direction
        hidden_states = hidden_states + self.alpha * direction
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register(self, model: nn.Module, layer_name: str):
        """
        Registers the hook on the specified layer.
        """
        for idx, (name, module) in enumerate(model.named_modules()):
            if name == layer_name:
                self.handle = module.register_forward_hook(self._hook)
                break

    def remove(self):
        """Removes the registered hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
