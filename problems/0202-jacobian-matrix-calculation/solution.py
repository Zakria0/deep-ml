import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	f_x = f(x)
	n = len(x)
	m = len(f_x)

	J = []

	for i in range(m):
		row = []
		for j in range(n):
			dfi_dxj = (f(x[:j]+[x[j]+h]+x[j+1:])[i] - f(x[:j]+[x[j]-h]+x[j+1:])[i]) / (2 * h)
			row.append(dfi_dxj)
		J.append(row)
	
	return J





