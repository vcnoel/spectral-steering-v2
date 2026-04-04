import torch

def extract_behavior_eigenvector(aligned_activations: torch.Tensor, misaligned_activations: torch.Tensor, return_all=False):
    """
    Extracts the primary behavior eigenvector from contrastive activation pairs.
    
    Args:
        aligned_activations: (n_pairs, hidden_dim)
        misaligned_activations: (n_pairs, hidden_dim)
        return_all: If True, returns full U, S, Vh
        
    Returns:
        u1: First left singular vector
        s1: First singular value
        v1: First right singular vector (the "behavior direction")
    """
    # 1. Compute difference vectors
    D = aligned_activations - misaligned_activations  # (n_pairs, hidden_dim)
    
    # Mean-center differences
    D_centered = D - D.mean(dim=0, keepdim=True)
    
    # 2. SVD
    # D is (N, d). Vh is (d, d), U is (N, d), S is (min(N,d))
    U, S, Vh = torch.linalg.svd(D_centered, full_matrices=False)
    
    if return_all:
        return U, S, Vh
        
    u1 = U[:, 0]
    s1 = S[0]
    v1 = Vh[0, :]
    
    return u1, s1, v1
