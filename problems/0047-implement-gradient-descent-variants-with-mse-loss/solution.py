import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    n, d = X.shape
    if method == 'batch':
        for epoch in range(n_epochs):
            grad = 2 * X.T @ (X @ weights - y) / n
            weights -= learning_rate * grad
        
    if method == 'stochastic':
        for epoch in range(n_epochs):
            for i in range(n):
                weights -= 2 * learning_rate * (X[i] * (X[i] @ weights - y[i]))
        
    if method == 'mini_batch':
        for epoch in range(n_epochs):
            for i in range(0, n, batch_size):
                X_batch, y_batch = X[i: i + batch_size], y[i: i + batch_size]
                weights -= 2 * learning_rate * X_batch.T @ (X_batch @ weights - y_batch) / X_batch.shape[0]
    
    return weights










