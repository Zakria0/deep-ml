import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	if func_name == 'poly2d':
		return (
			2 * point[0] * point[1] + point[1] ** 2,
			2 * point[1] * point[0] + point[0] ** 2
		)
	elif func_name == 'exp_sum':
		return (
			np.exp(point[0] + point[1]),
			np.exp(point[0] + point[1])
		)
	elif func_name == 'product_sin':
		return (
			np.sin(point[1]),
			point[0] * np.cos(point[1])
		)
	elif func_name == 'poly3d':
		return (
			2 * point[0] * point[1],
			point[0] ** 2 + point[2] ** 2,
			2 * point[1] * point[2]
		)
	else:
		return (
			2 * (point[0] - point[1]),
			-2 * (point[0] - point[1])
		)