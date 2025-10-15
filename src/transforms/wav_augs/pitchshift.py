from typing import Any

from torch import Tensor, nn
from torch_audiomentations import PitchShift


class ShiftPitch(nn.Module):
    """Pitch shift augmentation for waveform tensors.
    Args:
        sample_rate
    """

    def __init__(self, sample_rate: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._aug = PitchShift(sample_rate=sample_rate, output_type="dict")

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x)["samples"].squeeze(1)
