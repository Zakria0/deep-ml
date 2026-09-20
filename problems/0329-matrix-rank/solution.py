import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    M = np.array(A, dtype=float, copy=True)
    m, n = M.shape
    rank = 0
    row = 0
    
    for col in range(n):
        if row >= m:
            break
        
        # Partial pivoting: find the largest-magnitude entry in this column
        pivot = row + np.argmax(np.abs(M[row:, col]))
        
        # Column has no nonzero entries (within tol) -> contributes nothing
        if abs(M[pivot, col]) <= tol:
            continue
        
        # Swap pivot into position
        M[[row, pivot]] = M[[pivot, row]]
        
        # Eliminate below the pivot
        M[row] = M[row] / M[row, col]
        for i in range(row + 1, m):
            M[i] -= M[i, col] * M[row]
        
        rank += 1
        row += 1
    
    return rank