import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_errors, val_errors = [], []

    X_train = np.asarray(X_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    DM_val = np.array([[xi ** d for d in range(degree + 1)] for xi in X_val])
    DM_train_full = np.array([[xi ** d for d in range(degree + 1)] for xi in X_train])

    for n in train_sizes:
        DM_train = DM_train_full[:n]

        y_pred_train = DM_train @ (np.linalg.pinv(DM_train) @ y_train[:n])
        y_pred_val = DM_val @ (np.linalg.pinv(DM_train) @ y_train[:n])

        train_e = np.mean((y_pred_train - y_train[:n]) ** 2)
        val_e = np.mean((y_pred_val - y_val) ** 2)

        train_errors.append(train_e)
        val_errors.append(val_e)

    if train_errors[-1] > bias_threshold:
        diagnosis = 'high_bias'
    elif val_errors[-1] - train_errors[-1] > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
    
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnosis
    }