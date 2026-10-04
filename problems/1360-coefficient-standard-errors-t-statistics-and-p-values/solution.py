import math

import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    n, p = X.shape

    D = np.column_stack([np.ones(len(X)), X])

    coef = np.linalg.lstsq(D, y, rcond=None)[0]
    inv  = np.linalg.inv(D.T @ D)
    RSS  = np.sum((D @ coef - y) ** 2, keepdims=True)
    
    var = RSS / (n - p - 1)
    se  = np.diagonal(np.sqrt(var * inv))
    t   = coef / se
    p   = np.asarray([2 * (1 - 0.5 * (1 + math.erf(np.abs(tj) / np.sqrt(2)))) for tj in t])

    return coef, se, t, p


