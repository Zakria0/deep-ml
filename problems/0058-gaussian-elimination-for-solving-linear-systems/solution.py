import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    n = len(b)

    for col in range(n):
        pivot = col + np.argmax(np.abs(A[col:, col]))
        if abs(A[pivot, col]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]

        for i in range(col + 1, n):
            factor = A[i, col] / A[col, col]
            A[i, col:] -= factor * A[col, col:]
            b[i] -= factor * b[col]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

    return x