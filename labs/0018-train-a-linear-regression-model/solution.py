import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    method = 'normal'

    if method == 'normal':
        ones = np.ones((X.shape[0], 1))
        X_b = np.concatenate([ones, X], axis=1)
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        return theta[1:], theta[0]

    if method == "gradient":
        lr = 0.01
        num_it = 1000
        n = X.shape[0]
        
        for _ in range(num_it):
            preds = X @ W + b
            errors = preds - y
            grad_w = 2 / n * (X.T @ errors)
            grad_b = 2 / n * np.sum(errors)
            W = W - lr * grad_w
            b = b - lr * grad_b
        return W, b

    if method == "momentum":
        lr = 0.01
        beta = 0.9
        num_it = 3000
        n = X.shape[0]
        v_w = np.zeros(W.shape)
        v_b = 0

        
        for _ in range(num_it):
            preds = X @ W + b
            errors = preds - y
            grad_w = 2 / n * (X.T @ errors)
            grad_b = 2 / n * np.sum(errors)
            v_w = beta * v_w - lr * grad_w
            v_b = beta * v_b - lr * grad_b
            W = W + v_w
            b = b + v_b
        return W, b

