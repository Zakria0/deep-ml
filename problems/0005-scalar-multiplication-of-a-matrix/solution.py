def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	return [[scalar * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]