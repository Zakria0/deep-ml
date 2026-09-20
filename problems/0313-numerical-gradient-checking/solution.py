import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    n = len(x)
    numerical_gradient = []
    identity = np.eye(n)

    for i in range(n):
        e = identity[i] * epsilon
        df_xi = (f(x + e) - f(x - e)) / (2 * epsilon)
        numerical_gradient.append(df_xi)

    numerical_gradient = np.array(numerical_gradient)

    resid_norm = np.sqrt(sum((numerical_gradient - analytical_grad) ** 2))
    num_norm = np.sqrt(sum(numerical_gradient ** 2))
    analytical_norm = np.sqrt(sum(analytical_grad ** 2))

    if num_norm < epsilon and analytical_norm < epsilon:
        return epsilon
    
    err = resid_norm / (num_norm + analytical_norm)

    return numerical_gradient, err




