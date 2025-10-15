from einops import rearrange
from torch import nn

from .activation_and_ffn import GLUActivation, SwishActivation


class Conv(nn.Module):
    """Conformer depthwise separable convolution block.

    Args:
        model_dim: input dimension
        expansion_factor: expansion factor for the 1x1 conv
        p_dropout: dropout probability
        kernel_size: depthwise conv kernel size
    """

    def __init__(self, model_dim, expansion_factor, p_dropout, kernel_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(model_dim, model_dim * expansion_factor, kernel_size=1),
            GLUActivation(dim=1),
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=model_dim,
            ),
            nn.BatchNorm1d(model_dim),
            SwishActivation(),
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
            nn.Dropout(p_dropout),
        )

    def forward(self, input):
        input_tr = self.layer_norm(input).transpose(1, 2)
        output = self.sequential(input_tr)
        return output.transpose(1, 2)


class Conv2dSubsampling(nn.Module):
    """2D convolutional subsampling used before the conformer blocks.

    Input
    - inputs: tensor with shape (B, T, F)
    - inputs_lengths: tensor with shape (B,)

    Returns
    - outputs, output_lengths where outputs are subsampled features and output_lengths is there lengths
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs, inputs_lengths):
        outputs = self.subsampling(inputs.unsqueeze(1))
        outputs = rearrange(outputs, "b c l d -> b l (d c)")
        output_lengths = (inputs_lengths >> 2) - 1
        return outputs, output_lengths
