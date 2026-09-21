import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    x = torch.flatten(x).reshape(new_shape)
    return x

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(x, -1, -2)
