import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	def derive(func, x):
		if func == 'square':
			return 2 * x
		elif func == 'sin':
			return np.cos(x)
		elif func == 'exp':
			return np.exp(x)
		else:
			return 1 / x if x else None
	
	def evaluate(func, x):
		if func == 'square':
			return x ** 2
		elif func == 'sin':
			return np.sin(x)
		elif func == 'exp':
			return np.exp(x)
		else:
			return np.log(x) if x else None
	
	d_gx = derive(functions[-1], x)

	if len(functions) == 1:
		return d_gx

	gx = evaluate(functions[-1], x)

	return compute_chain_rule_gradient(functions[:-1], gx) * d_gx
