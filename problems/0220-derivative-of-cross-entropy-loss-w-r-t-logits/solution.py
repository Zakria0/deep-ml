def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	from math import exp
	sm = sum(exp(i) for i in logits)
	p = [exp(i) / sm for i in logits]
	return [p[i] - int(i == target) for i in range(len(logits))]