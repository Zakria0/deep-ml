def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	parameters_new = parameters[:]
	for i in range(len(parameters_new)):
		for j in range(len(parameters_new[0])):
			parameters_new[i][j] -= lr * (gradients[i][j] + weight_decay * int(apply_to_all[i]) * parameters_new[i][j])

	return parameters_new