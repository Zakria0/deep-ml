import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    def gini(y):
        counts = {}
        n = len(y)
        for c in y:
            counts[c] = counts.get(c, 0) + 1
        return  1 - np.sum([(counts[i] / n) ** 2 for i in counts])

    n, n_features = X.shape
    best_gini = np.inf
    feature_index, threshold = -1, None
    for f in range(n_features):
        sort = np.argsort(X[:, f])
        X_sorted, y_sorted = X[sort, f], y[sort]
        for i in range(1, n):
            if X_sorted[i] == X_sorted[i - 1]:
                continue
            y_L, y_R = y_sorted[:i], y_sorted[i:]
            g_score = len(y_L) / n * gini(y_L) + len(y_R) / n * gini(y_R)
            
            if g_score < best_gini:
                best_gini = g_score
                feature_index = f
                threshold = X_sorted[i-1]
    
    return feature_index, threshold
