import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    rng = np.random.default_rng(seed=seed)
    n, d = X_train.shape

    preds = np.empty((n_estimators, len(X_test)), dtype=int)
    for b in range(n_estimators):
        idx = rng.choice(n, size=n)
        Xb, yb = X_train[idx], y_train[idx]

        best_e, j_opt, t_opt, flip = np.inf, None, None, 0
        for j in range(d):
            for t in np.unique(Xb[:, j]):
                p = (Xb[:, j] > t).astype(int)
                for cand, fl in ((p, 0), (1 - p, 1)):
                    e = int(np.sum(cand != yb))
                    if e < best_e:
                        best_e, j_opt, t_opt, flip = e, j, t, fl

        pred = (X_test[:, j_opt] > t_opt).astype(int)
        preds[b] = 1 - pred if flip else pred

    votes = preds.sum(axis=0)
    return (votes * 2 >= n_estimators).astype(int)
