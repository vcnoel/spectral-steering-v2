import torch

def apply_rank1_deflation(weight_matrix: torch.Tensor, v_behavior: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies rank-1 spectral deflation to a weight matrix.
    
    Args:
        weight_matrix: The original weight matrix (out_features, in_features). e.g., mlp.down_proj.
        v_behavior: The behavior eigenvector (right singular vector of diffs). Shape (out_features,).
        alpha: Deflation strength.
        
    Returns:
        W_new: The deflated weight matrix.
    """
    W_old_device = weight_matrix.device
    v = v_behavior.to(W_old_device).squeeze()
    
    # We want to remove the component of W_old's output that aligns with v.
    # W_old is [out_features, in_features]. v is [out_features].
    # Projection of W columns onto v: v @ W_old -> shape [in_features]
    proj = torch.matmul(v, weight_matrix) 
    
    # W_new = W_old - alpha * (v v^T) W_old = W_old - alpha * outer(v, v^T W_old)
    rank1_update = alpha * torch.outer(v, proj)
    
    W_new = weight_matrix - rank1_update
    return W_new

def deflate_module_weight(module: torch.nn.Module, v_behavior: torch.Tensor, alpha: float, sigma: float = None):
    """
    In-place deflation of a PyTorch linear module using a behavior direction.
    """
    W_old = module.weight.data
    W_new = apply_rank1_deflation(W_old, v_behavior, alpha)
    module.weight.data = W_new.to(module.weight.dtype)
