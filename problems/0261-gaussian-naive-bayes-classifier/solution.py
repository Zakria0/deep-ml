import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
	classes = np.unique(y_train)
	n_samples, n_features = X_train.shape
	n_classes = len(classes)
	priors = np.zeros(n_classes)
	means, variances = np.zeros((n_classes, n_features)), np.zeros((n_classes, n_features))

	for i, k in enumerate(classes):
		Xk = X_train[y_train == k]
		priors[i] = len(Xk) / n_samples
		means[i] = np.mean(Xk, axis=0)
		variances[i] = np.var(Xk, axis=0) + 1e-9
	
	log_priors = np.log(priors)
	preds = np.zeros(X_test.shape[0])
	for i, x in enumerate(X_test):
		log_likelihood = -0.5 * (np.log(2 * np.pi * variances) + (x - means)**2 / variances)
		preds[i] = classes[np.argmax(log_priors + np.sum(log_likelihood, axis=1))]
	
	return np.int32(preds)


		






