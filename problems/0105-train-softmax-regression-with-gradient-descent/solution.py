import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	n, d = X.shape
	C = y.max() + 1
	M = d + 1
	X = np.hstack([np.ones((n, 1)), X])

	weights = np.zeros((C, M))
	losses = []

	for _ in range(iterations):
		z = X @ weights.T
		p = np.exp(z)
		p /= p.sum(axis=1, keepdims=True)

		loss = -np.log(p[np.arange(n), y]).sum()
		losses.append(loss)

		g = p.copy()
		g[range(n), y] -= 1.

		weights -= learning_rate * (g.T @ X)
	
	return weights, losses
		
		
		
		
		
		
		
		
		
		
		
