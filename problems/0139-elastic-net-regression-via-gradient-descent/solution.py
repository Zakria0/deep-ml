import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n, p = X.shape
    w, b = np.zeros(p), 0
    L1 = np.inf
    i = 0

    while L1 > tol and i < max_iter:
        y_pred = X @ w + b
        e = y_pred - y
        g_w, g_b = 1 / n * X.T @ e + alpha1 * np.sign(w) + 2 * alpha2 * w, 1 / n * np.sum(e)
        w -= learning_rate * g_w
        b -= learning_rate * g_b
        L1 = np.sum(np.abs(g_w))
        i += 1
    return (w, b)









