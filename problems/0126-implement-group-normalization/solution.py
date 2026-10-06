import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    B, C, H, W = X.shape

    group_size = C // num_groups
    X = np.reshape(X, (B, num_groups, group_size, H, W))

    mean = np.mean(X, axis=(2, 3, 4), keepdims=True)
    var = np.var(X, axis=(2, 3, 4), keepdims=True)

    normalized_output = gamma * (np.reshape((X - mean) / np.sqrt((var + epsilon)), (B, C, H, W))) + beta

    return normalized_output