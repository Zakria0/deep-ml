import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n_samples, n_features = X.shape

    classes = np.unique(y)
    train, test = [], []
    
    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * test_size)
        train.extend(idx[split:])
        test.extend(idx[:split])
    
    train = np.asarray(train)
    test  =  np.asarray(test)

    np.random.shuffle(train)
    np.random.shuffle(test)

    X_train, y_train = X[train], y[train]
    X_test, y_test   =  X[test], y[test]

    return X_train, X_test, y_train, y_test

