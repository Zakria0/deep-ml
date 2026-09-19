import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    X1sq = np.sum(X1**2, axis=1, keepdims=True)
    X2sq = np.sum(X2**2, axis=1, keepdims=True)

    d = X1sq + X2sq.T - 2 * X1 @ X2.T

    return np.exp(-gamma * d)