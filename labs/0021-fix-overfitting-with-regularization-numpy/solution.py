import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    
    #1. PCA
    n, d = X_train.shape
    mean       = np.mean(X_train, axis=0)
    X_centered = X_train - mean

    mat = np.cov(X_centered, rowvar=False)
    eig_val, eig_vec = np.linalg.eigh(mat)
    
    idx     = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, idx]
    eig_val = eig_val[idx]

    s, i = 0, 0
    tot = np.sum(eig_val)
    pc   = []
    while s < 0.9 and i < d:
        pc.append(eig_vec[:, i])
        s += eig_val[i] / tot
        i += 1

    pc         = np.column_stack(pc)
    X_train_pc = np.dot(X_centered, pc)
    X_val_pc   = np.dot(X_val - mean, pc)

    #2. L2 + early stopping

    k = pc.shape[1]
    weight, b = np.zeros(k), 0.0
    lr, alpha = 0.01, 0.01
    patience, min_delta = 10, 1e-4

    best_val = np.inf
    best = (weight.copy(), b)
    counter = 0

    for _ in range(20000):
        pred = X_train_pc @ weight + b
        e = pred - y_train

        grad_w = X_train_pc.T @ e / n + 2 * alpha * weight
        grad_b = e.mean()

        weight -= lr * grad_w
        b      -= lr * grad_b

        val_pred = X_val_pc @ weight + b
        val_loss = np.mean((val_pred - y_val) ** 2)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best = (weight.copy(), b)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    w, b = best

    def predict(X):
        return ((X - mean) @ pc) @ w + b
    return predict

    """
    ybar = y_train.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (y_train - ybar))
    b = ybar
    """
