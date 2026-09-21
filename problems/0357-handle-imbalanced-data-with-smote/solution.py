import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)

    if not k_actual or not n_synthetic:
        return np.empty((0, n_features))
    
    X_synthetic = []
    for _ in range(n_synthetic):
        i   = np.random.randint(0, n_samples)
        x_i = X_minority[i]
        
        dist    = np.sum((X_minority - x_i) ** 2, axis=1)
        knn     = np.argsort(dist)[1: k_actual+1]
        x_nn    = X_minority[knn[np.random.randint(0, k_actual)]]
        x_synth = x_i + np.random.random() * (x_nn - x_i)

        X_synthetic.append(x_synth)
    
    return X_synthetic








