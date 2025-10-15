from math import log

import torch
from torch import nn


class RelativePosEmb(nn.Module):
    """Relative positional embedding.

    Args:
        model_dim: embedding dimensionality (D)

    Input Shape
    - x: tensor with shape (B, L, D)

    Returns
    - tensor with shape (B, L, D) containing positional embeddings
    """

    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim

    def forward(self, x):
        relative_ind = (
            torch.arange(0, x.shape[1], dtype=torch.float32).unsqueeze(1).to(x.device)
        )
        relative_ind = relative_ind * torch.exp(
            -torch.arange(0, 2 * x.shape[2], 2, dtype=torch.float32).to(x.device)
            * (log(10000.0) / self.model_dim)
        ).unsqueeze(0)
        relative_ind[:, ::2] = torch.cos(relative_ind[:, ::2])
        relative_ind[:, 1::2] = torch.sin(relative_ind[:, 1::2])
        return relative_ind.repeat(x.shape[0], 1, 1)
