import torch

def linear_backward(grad_output, x, W):
    return grad_output @ W, grad_output.T @ x, grad_output.sum(axis=0)