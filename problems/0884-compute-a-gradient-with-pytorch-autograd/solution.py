import torch

def grad_of_quadratic(x_value: float) -> float:
    x = torch.tensor(x_value, requires_grad=True)
    f_x = x**2 + 3 * x + 2
    f_x.backward()
    return float(x.grad)