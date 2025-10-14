from torch import Tensor, nn
from audiomentations import AddBackgroundNoise, PolarityInversion

class AddNoise(nn.Module):
    def __init__(self,sounds_path, sample_rate, *args, **kwargs):
        super().__init__()
        self.sample_rate=sample_rate
        self._aug = AddBackgroundNoise(
            sounds_path,
            noise_transform=PolarityInversion(),
            p=0.5,
            *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.squeeze(0)
        return self._aug(x, sample_rate=self.sample_rate).unsqueeze(0)
