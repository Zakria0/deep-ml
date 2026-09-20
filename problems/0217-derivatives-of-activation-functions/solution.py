def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	from math import expm1

	sig_x = 1 / (1 + (expm1(-x) + 1))
	d_sig = sig_x * (1 - sig_x)

	tanh_x = (expm1(x) - expm1(-x)) / (expm1(x) + expm1(-x) + 2)
	d_tanh = 1 - tanh_x ** 2

	d_relu = 1 if x > 0 else 0

	return {'sigmoid': d_sig, 'tanh': d_tanh, 'relu': d_relu}