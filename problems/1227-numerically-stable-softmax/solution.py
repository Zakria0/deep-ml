import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    maxdim = torch.max(t, dim=dim, keepdim=True).values
    return torch.exp(t - maxdim) / torch.sum(torch.exp(t - maxdim), keepdim=True, dim=dim)