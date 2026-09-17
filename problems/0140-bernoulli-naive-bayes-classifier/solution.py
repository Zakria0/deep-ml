import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        self.classes_ = np.unique(y)
        
        n_features, n_classes = X.shape[1], len(self.classes_)
        if n_classes == 1:
            self.flag_ = True
            self.label_ = self.classes_[0]
            return self
        self.flag_ = False
        self.priors_, self.p_ = np.zeros(n_classes), np.zeros((n_classes, n_features))
        for i, k in enumerate(self.classes_):
            Xk = X[y == k]
            nk = len(y[y == k])
            self.priors_[i] = nk / len(y)
            ck = Xk.sum(axis=0)
            self.p_[i] = (ck + self.smoothing) / (nk + 2 * self.smoothing)
        
        self.logpriors_ = np.log(self.priors_)
        self.logp_ = np.log(self.p_)
        self.log1mp_ = np.log(1 - self.p_)

        return self

    def predict(self, X):
        # Predict class labels for test set X
        if self.flag_:
            return np.array([self.label_] * X.shape[0])
        
        log_likelihoods = (
            self.logpriors_ +
            X @ self.logp_.T +
            (1 - X) @ self.log1mp_.T
        )
        return self.classes_[np.argmax(log_likelihoods, axis=1)]
        




