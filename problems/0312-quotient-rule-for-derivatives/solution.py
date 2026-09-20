import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    h_degree = len(h_coeffs) - 1
    g_degree = len(g_coeffs) - 1

    h_x  = sum(h_coeffs[i] * x ** (h_degree - i) for i in range(h_degree + 1))
    hd_x = sum((h_degree - i) * h_coeffs[i] * x ** (h_degree - i - 1) for i in range(h_degree))

    g_x  = sum(g_coeffs[i] * x ** (g_degree - i) for i in range(g_degree + 1))
    gd_x = sum((g_degree - i) * g_coeffs[i] * x ** (g_degree - i - 1) for i in range(g_degree))

    return (gd_x * h_x - g_x * hd_x) / (h_x ** 2)