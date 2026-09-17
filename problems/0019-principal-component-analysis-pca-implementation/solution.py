import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    X = (data - mean) / std

    cov_mat = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat, UPLO='L')
    idx = np.argsort(eigenvalues)[::-1]
    for j in range(eigenvectors.shape[1]):
        col = eigenvectors[:, j] 
        for elem in col:
            if np.abs(elem) > 1e-10:
                if elem < 0:
                    eigenvectors[:, j] *= -1 
                break
    
    return eigenvectors[:, idx[:k]]




