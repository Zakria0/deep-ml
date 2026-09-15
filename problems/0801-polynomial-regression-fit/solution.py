import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    X = np.array([[xi ** d for d in range(degree + 1)] for xi in x])
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    return [float(c) for c in coefs]