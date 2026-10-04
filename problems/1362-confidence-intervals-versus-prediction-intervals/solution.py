import numpy as np


def interval_halfwidths(X: np.ndarray, y: np.ndarray, x0: np.ndarray, z: float) -> tuple:
    """Half-widths of the mean-response and new-observation intervals at x0.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.
        x0 (np.ndarray): (p,) point at which to form the intervals.
        z (float): multiplier, e.g. 1.96 for approximately 95%.

    Returns:
        tuple: (ci_half, pi_half) as floats.
    """
    n, p = X.shape

    D  = np.column_stack([np.ones(n), X])
    d0 = np.concatenate([[1.0], x0])
    
    coef = np.linalg.lstsq(D, y, rcond=None)[0]
    inv  = np.linalg.inv(D.T @ D)
    RSS  = np.sum((D @ coef - y)**2)

    sigma = RSS / (n - p - 1)
    var   = sigma * d0.T @ inv @ d0

    return z * np.sqrt(var), z * np.sqrt(sigma + var)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
