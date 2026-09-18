import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)

    X = X - mean
    cov_mat = X.T @ X / n_samples
    eig_val, eig_vec = np.linalg.eigh(cov_mat, UPLO='L')

    idx = np.argsort(eig_val)[::-1]#[:n_components:-1]
    return np.sum(eig_val[idx[n_components:]]) / n_features

    pc_vec = eig_vec[:, idx]
    proj = X @ pc_vec
    rec = proj @ pc_vec.T

    MSE = np.mean((X - (rec + mean)) ** 2)

    return MSE






