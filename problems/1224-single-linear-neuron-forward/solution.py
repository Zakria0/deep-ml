import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    ln = nn.Linear(3, 1)

    with torch.no_grad():
        weight = torch.tensor([[0.5, -0.2, 0.3]])
        bias   = torch.tensor([0.1])

        ln.weight.copy_(weight)
        ln.bias.copy_(bias)

    return ln(x).item()