import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    n = len(X[0])
    mean = np.mean(X, axis=0)

    X = X - mean
    cov_matrix = (X.T @ X) / (n - 1)

    eig_val = np.linalg.eigvals(cov_matrix)
    ratios = sorted(eig_val / sum(eig_val))[::-1]

    return ratios







