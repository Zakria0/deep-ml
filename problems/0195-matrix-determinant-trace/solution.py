def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code here
	tr = sum(matrix[i][i] for i in range(len(matrix)))

	def det(matrix):
		if len(matrix) == 2:
			return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

		s = 0
		for i in range(len(matrix)):
			ai0 = (-1) ** i * matrix[i][0]
			cof_mat = [[matrix[k][l] for l in range(1, len(matrix[0]))] for k in range(len(matrix)) if k != i]
			cof = det(cof_mat)
			s += ai0 * cof

		return s
	
	determinant = det(matrix)

	return determinant, tr