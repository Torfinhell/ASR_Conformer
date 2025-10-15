from typing import Any

import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    """Gain augmentation for waveform tensors."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._aug = torch_audiomentations.Gain(*args, **kwargs, output_type="dict")

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        return self._aug(x)["samples"].squeeze(1)
