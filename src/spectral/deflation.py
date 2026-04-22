import torch

def apply_rank_k_deflation(weight_matrix: torch.Tensor, v_behavior_k: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies rank-k spectral deflation to a weight matrix.
    
    Args:
        weight_matrix: [out_features, in_features]
        v_behavior_k: [out_features, k] - k behavior eigenvectors (singular vectors)
        alpha: Deflation strength.
        
    Returns:
        W_new: The deflated weight matrix.
    """
    W_old_device = weight_matrix.device
    V = v_behavior_k.to(W_old_device) # [out, k]
    
    # 1. Project weight rows onto behavior subspace
    # proj = V^T @ W -> [k, in_features]
    proj = torch.matmul(V.t(), weight_matrix)
    
    # 2. Reconstruct the behavioral component
    # upgrade = V @ proj -> [out, in_features]
    rank_k_update = alpha * torch.matmul(V, proj)
    
    W_new = weight_matrix - rank_k_update
    return W_new

def deflate_module_weight(module: torch.nn.Module, v_behavior: torch.Tensor, alpha: float):
    """
    In-place deflation of a PyTorch linear module.
    v_behavior can be a single vector [out] or a matrix [out, k].
    """
    W_old = module.weight.data
    V = v_behavior
    if V.dim() == 1:
        V = V.unsqueeze(1)
    
    W_new = apply_rank_k_deflation(W_old, V, alpha)
    module.weight.data = W_new.to(module.weight.dtype)
