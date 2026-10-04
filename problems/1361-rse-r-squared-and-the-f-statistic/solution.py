import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    n, p = X.shape

    D = np.column_stack([np.ones(n), X])

    coef = np.linalg.lstsq(D, y, rcond=None)[0]
    mean = np.mean(y)

    RSS = np.sum((D @ coef - y) ** 2)
    TSS = np.sum((y - mean) ** 2)

    if RSS < 1e-9:
        return 0.0, 1.0, float('inf')

    r2 = 1 - RSS / TSS
    rse = np.sqrt(RSS / (n - p - 1))
    f_stat = (TSS - RSS) / p / (RSS / (n - p - 1))

    return float(rse), float(r2), float(f_stat)
    

