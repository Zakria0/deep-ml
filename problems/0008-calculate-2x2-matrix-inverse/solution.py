def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    if not det:
        return None

    return [[matrix[1][1] / det, - matrix[0][1] / det], [-matrix[1][0] / det, matrix[0][0] / det]]