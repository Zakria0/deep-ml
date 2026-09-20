def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	m, n = len(matrix), len(matrix[0])
	if mode == 'column':
		for i in range(n):
			mean = 0
			for j in range(m):
				mean += matrix[j][i]
			means.append(mean / m)

	else:
		for i in range(m):
			mean = 0
			for j in range(n):
				mean += matrix[i][j]
			means.append(mean / n)
	
	return means