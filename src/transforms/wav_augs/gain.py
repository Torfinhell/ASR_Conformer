import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = torch_audiomentations.Gain(*args, **kwargs,output_type='dict')

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x)['samples'].squeeze(1)
