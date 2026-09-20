def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    if not n:
        return 0
    
    return c * n * x ** (n-1)