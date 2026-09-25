import sys
import numpy as np


class DecisionTree:
    """
    A decision tree classifier that the harness will use as the base learner
    in a Random Forest. The harness will create many DecisionTree instances,
    train each on a different bootstrap sample of the data, and aggregate
    their predictions by majority vote.

    For the ensemble to beat a single tree by a meaningful margin, your trees
    must be DIVERSE. Bootstrap sampling (handled by the harness) gives some
    diversity. The most effective additional source of diversity is to
    randomize WHICH X each split considers -- this is what makes a
    Random Forest different from plain Bagging.

    Use `self.random_state` for any randomness inside your tree so that
    different harness seeds produce different trees.
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 max_features=16, splitter='random', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features      
        self.random_state = random_state
        self.splitter = splitter

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - ((counts / counts.sum()) ** 2).sum()

    def _split_dataset(self, X, y, j, t):
        mask = X[:, j] <= t
        return X[mask], y[mask], X[~mask], y[~mask]

    def _leaf_prediction(self, y):
        return np.bincount(y).argmax()

    def _best_split(self, X, y, feats):
        n, K = X.shape[0], self.n_classes_
        counts = np.bincount(y, minlength=K)
        g_parent = 1.0 - ((counts / n) ** 2).sum()
        best_gain, best = 0.0, None
        for j in feats:
            order = np.argsort(X[:, j])
            xs, ys = X[order, j], y[order]
            cum = np.cumsum(ys[:, None] == np.arange(K), axis=0)
            valid = xs[:-1] < xs[1:]
            nL = np.arange(1, n)
            left, right = cum[:-1], counts - cum[:-1]
            gL = 1.0 - ((left  / nL[:, None])**2).sum(axis=1)
            gR = 1.0 - ((right / (n - nL)[:, None]) ** 2).sum(axis=1)
            gain = g_parent - (nL * gL + (n - nL) * gR) / n
            gain[~valid] = -1.0
            t = int(np.argmax(gain))
            if gain[t] > best_gain:
                best_gain, best = gain[t], (j, (xs[t] + xs[t + 1]) / 2.0)
        return best
    
    def _best_split_random(self, X, y, feats):
        n, K = X.shape[0], self.n_classes_
        counts = np.bincount(y, minlength=K)
        g_parent = 1.0 - ((counts / n) ** 2).sum()
        best_gain, best = 0.0, None
        for j in feats:
            col = X[:, j]
            lo, hi = col.min(), col.max()
            if lo == hi:
                continue
            t = lo + self.rng.random_sample() * (hi - lo)
            mask = col <= t
            nL = int(mask.sum())
            if nL == 0 or nL == n:
                continue
            cL = np.bincount(y[mask], minlength=K)
            cR = counts - cL
            gL = 1.0 - ((cL / nL) ** 2).sum()
            gR = 1.0 - ((cR / (n - nL)) ** 2).sum()
            gain = g_parent - (nL * gL + (n - nL) * gR) / n
            if gain > best_gain:
                best_gain, best = gain, (j, float(t))
        return best
    
    def _best_split_fully_random(self, X, y):
        j = self.rng.randint(X.shape[1])
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return None
        t = lo + self.rng.random_sample() * (hi - lo)
        return (j, float(t)) 

    def _build(self, X, y, depth=0):
        if (len(np.unique(y)) == 1 
        or len(y) < self.min_samples_split
        or (self.max_depth is not None and depth >= self.max_depth)):
            return {'leaf': True, 'prediction': self._leaf_prediction(y)}

        if self.splitter == 'fully_random':
            best = self._best_split_fully_random(X, y)
        else:
            feats = self.rng.choice(X.shape[1], size=self.m_, replace=False)
            best = (self._best_split_random(X, y, feats) if self.splitter == 'random'
                    else self._best_split(X, y, feats))

        if best is None:
            return {'leaf': True, 'prediction': self._leaf_prediction(y)}
        j, t = best
        Xl, yl, Xr, yr = self._split_dataset(X, y, j, t)
        return {'leaf': False, 'feature_index': j, 'threshold': t,
                'left':  self._build(Xl, yl, depth + 1),
                'right': self._build(Xr, yr, depth + 1)}

    def fit(self, X, y): 
        sys.setrecursionlimit(10000)
        y = y.astype(int)
        self.n_classes_ = int(y.max()) + 1
        self.rng = np.random.RandomState(self.random_state)
        d = X.shape[1]
        self.m_ = (int(np.sqrt(d)) if self.max_features == 'sqrt'
                   else d if self.max_features is None else int(self.max_features))
        self.m_ = min(self.m_, X.shape[1])        
        self.root_ = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        out = np.empty(X.shape[0], dtype=int)
        stack = [(self.root_, np.arange(X.shape[0]))]
        while stack:
            node, ids = stack.pop()
            if node['leaf']:
                out[ids] = node['prediction']
            else:
                go_left = X[ids, node['feature_index']] <= node['threshold']
                stack.append((node['left'],  ids[go_left]))
                stack.append((node['right'], ids[~go_left]))
        return out
