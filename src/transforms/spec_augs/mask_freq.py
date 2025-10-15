import torch_audiomentations
from torch import Tensor, nn
import torch

class MaskFreq(nn.Module):
    def __init__(self, F=27, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F=F
    def __call__(self, data: Tensor):
        spectrogram=data.clone()
        start=torch.randint(0, spectrogram.shape[1]-self.F, (1,)).item()
        spectrogram[:,start:start+self.F, :]=0
        return spectrogram

