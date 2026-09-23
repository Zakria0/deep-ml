import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV, RidgeCV, LassoCV
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: LinearRegression() WILL overfit here.
    - LinearRegression(): Train R² ≈ 1.0, Val R² ≈ -5.0
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
    # TODO: fit a regularized model and return a predict function
    # 
    # Ideas:
    #   - Ridge(alpha=??) or RidgeCV (auto-tunes alpha)
    #   - Lasso(alpha=??) or LassoCV (auto-tunes + feature selection)
    #   - ElasticNet (combines L1 + L2)
    #   - SelectKBest + Ridge pipeline
    #   - Any sklearn approach that handles overfitting!
    #
    # Plain LinearRegression() will NOT work here.
    
    models = [
        RidgeCV(alphas=np.logspace(-3, 4, 40)),
        LassoCV(n_alphas=50, cv=5),
        ElasticNetCV(l1_ratio=np.linspace(0.01, 0.99, 10),
                 n_alphas=50, cv=5)
    ]

    def R2(model, X, y):
        res = ((y - model.predict(X))**2).sum()
        tot = ((y - y.mean())**2).sum()
        return 1 - res / tot

    best_model, best_score = None, -np.inf
    for model in models:
        model.fit(X_train, y_train)
        score = R2(model, X_val, y_val)
        if score > best_score:
            best_model = model
            best_score = score

    def predict(X):
        return best_model.predict(X)

    return predict

    





