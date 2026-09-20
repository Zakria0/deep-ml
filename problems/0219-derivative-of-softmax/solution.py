def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	from math import expm1
	denom = sum((expm1(i) + 1) for i in x)
	s = [(expm1(i) + 1) / denom for i in x]

	n = len(x)
	J = []
	for i in range(n):
		row = []
		for j in range(n):
			row.append(s[i] * (int(i == j) - s[j]))
		J.append(row)

	return J






