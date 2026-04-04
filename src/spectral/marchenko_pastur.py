import numpy as np
import torch

def marchenko_pastur_threshold(S: torch.Tensor, n_samples: int, n_features: int, var: float = 1.0) -> torch.Tensor:
    """
    Applies Marchenko-Pastur thresholding to singular values.
    Returns a mask of singular values (and corresponding vectors) that are considered "signal".
    
    Args:
        S: Singular values (1D tensor)
        n_samples: Number of contrastive pairs
        n_features: Hidden dimension size
        var: Variance of the noise distribution
    
    Returns:
        Boolean tensor mask indicating which singular values exceed the MP noise upper bound.
    """
    # gamma = d / n (features / samples)
    # S are singular values, so eigenvalues of covariance are roughly S^2 / n_samples
    
    gamma = n_features / max(n_samples, 1)
    
    # Upper bound of the MP distribution support for covariance eigenvalues
    lambda_max = var * (1 + np.sqrt(gamma))**2
    
    # Convert singular values to equivalent covariance eigenvalues
    eigenvalues = (S ** 2) / max(n_samples, 1)
    
    # Mask of structural signal
    signal_mask = eigenvalues > lambda_max
    
    return signal_mask

def filter_noise_svd(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, n_samples: int, n_features: int, var: float = 1.0):
    """
    Filters the SVD components to only retain the ones above the Marchenko-Pastur threshold.
    """
    mask = marchenko_pastur_threshold(S, n_samples, n_features, var)
    return U[:, mask], S[mask], Vh[mask, :]
