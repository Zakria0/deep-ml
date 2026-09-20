import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	mag = np.sqrt(sum(g * g for g in gradient))

	if mag == 0:
		return {'magnitude': mag, 'direction': np.array([0] * len(gradient)), 'descent_direction': np.array([0] * len(gradient))}
	
	d = np.array(gradient) / mag

	return {'magnitude': mag, 'direction': d, 'descent_direction': -d}