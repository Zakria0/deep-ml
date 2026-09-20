import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if not vectors:
        return True
    
    M = np.array(vectors, dtype=float)
    m, n = M.shape
    tol = 1e-10
    rank = 0
    row = 0
    
    for col in range(n):
        if row >= m:
            break
        pivot = row + np.argmax(np.abs(M[row:, col]))
        if abs(M[pivot, col]) <= tol:
            continue
        M[[row, pivot]] = M[[pivot, row]]
        M[row] = M[row] / M[row, col]
        for i in range(row + 1, m):
            M[i] -= M[i, col] * M[row]
        rank += 1
        row += 1
    
    return rank == m