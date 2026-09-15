import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)

    means = np.mean(predictions, axis=0)
    bias2 = np.mean((means - y_true) ** 2)
    variance = np.mean(np.mean((predictions - means) ** 2, axis=0))
    MSE = np.mean(np.mean((predictions - y_true) ** 2, axis=0))

    return {
        'bias_squared': bias2,
        'variance': variance,
        'mse': MSE
    }