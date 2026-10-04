import torch
import torch.nn as nn


class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        # TODO: Build at least two hidden blocks:
        #   Linear -> BatchNorm1d -> ReLU (or similar) -> Dropout
        # Final layer: Linear to 1 logit.
        # Store layers on self (Sequential is fine).
        dim1 = hidden_dim
        dim2 = max(1, hidden_dim // 2)
        dim3 = max(1, hidden_dim // 4)
        
        block1 = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        block2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        block3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.BatchNorm1d(dim3),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

        self.model = nn.Sequential(
            block1,
            block2,
            block3,
            nn.Linear(dim3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        # TODO: run x through your network and squeeze the last dim if needed
        return self.model(x).squeeze(-1)


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # TODO:
    # - criterion = nn.BCEWithLogitsLoss()
    # - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # - loop epochs: zero_grad -> forward -> loss -> backward -> step
    # - y_train is float 0/1 with shape (N,)
    torch.manual_seed(42)

    model.train()

    n, d = X_train.shape

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y_train = y_train.float()

    for _ in range(epochs):
        optimizer.zero_grad()

        pred = model(X_train)
        loss = criterion(pred, y_train)

        loss.backward()

        optimizer.step()

    return model

