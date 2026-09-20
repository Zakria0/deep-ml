import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	H_P = 0
	H_PQ = 0

	for i in range(len(P)):
		H_P -= P[i] * np.log(P[i] + 1e-10)
		H_PQ -= P[i] * np.log(Q[i] + 1e-10)

	return H_P, H_PQ

