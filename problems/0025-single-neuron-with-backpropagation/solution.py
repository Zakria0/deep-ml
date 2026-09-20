import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	n, d = features.shape

	updated_weights = initial_weights.astype(float).copy()
	updated_bias = initial_bias
	mse_values = []

	for _ in range(epochs):
		z = np.dot(updated_weights, features.T) + updated_bias

		sig = 1 / (1 + np.exp(-z))
		
		MSE = 0
		for i in range(n):
			sig_zi = sig[i]
			resid = sig_zi - labels[i]
			MSE += resid ** 2
		MSE /= n

		D = []
		for j in range(d):
			sm = 0.0
			for i in range(n):
				sig_zi = sig[i]
				d_sig = sig_zi * (1 - sig_zi)
				sm += np.dot((sig_zi - labels[i]), d_sig) * features[i][j]
			D.append(2 * sm / n)
		
		dmse_db = 0
		for i in range(n):
			sig_zi = sig[i]
			d_sig = sig_zi * (1 - sig_zi)
			dmse_db += np.dot((sig_zi - labels[i]), d_sig)
		dmse_db = 2 * dmse_db / n

		updated_weights -= learning_rate * np.array(D)
		updated_bias -= learning_rate * dmse_db
		mse_values.append(MSE)



	return updated_weights, updated_bias, mse_values