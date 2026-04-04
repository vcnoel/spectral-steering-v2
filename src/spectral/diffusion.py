import torch
from spectral_trust.config import GSPConfig
from spectral_trust.graph import GraphConstructor

def apply_causal_heat_kernel_diffusion(activations: torch.Tensor, attention_matrix: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Applies graph diffusion via the causal heat kernel (I - tau * L).
    
    Args:
        activations: Node features / hidden states   [seq_len, hidden_dim]
                     or batched [batch, seq_len, hidden_dim]
        attention_matrix: Raw attention matrix       [batch, num_heads, seq_len, seq_len]
        tau: Diffusion time parameter (smoothing strength)
        
    Returns:
        Smoothed activations.
    """
    config = GSPConfig(symmetrization="symmetric", normalization="sym", remove_self_loops=True)
    graph_constructor = GraphConstructor(config)
    
    # Aggregate heads to [batch, seq_len, seq_len]
    adj = graph_constructor.aggregate_heads(attention_matrix)
    sym_adj = graph_constructor.symmetrize_attention(adj)
    
    # Construct normalized laplacian [batch, seq_len, seq_len]
    laplacian = graph_constructor.construct_laplacian(sym_adj)
    
    # Handle unbatched activations
    is_unbatched = False
    if activations.dim() == 2:
        activations = activations.unsqueeze(0)
        is_unbatched = True
        
    batch, seq_len, hidden_dim = activations.shape
    
    # I - tau * L
    I = torch.eye(seq_len, device=activations.device, dtype=activations.dtype).unsqueeze(0)
    diffusion_operator = I - tau * laplacian.to(activations.dtype)
    
    # Smooth: operator @ activations
    smoothed = torch.bmm(diffusion_operator, activations)
    
    if is_unbatched:
        return smoothed.squeeze(0)
    return smoothed
