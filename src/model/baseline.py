from torch import nn


class BaseLineModel(nn.Module):
    """A simple baseline ffn used for basic comparison.

    Input:
    - spectrogram: tensor with shape (B, F, T)
    - spectrogram_lengths: tensor with shape (B,)

    Returns
    - dict with keys log_probs and log_probs_length
    """

    def __init__(self, model_dim, n_tokens, fc_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_tokens),
        )

    def forward(self, spectrogram, spectrogram_lengths, **batch):
        output_net = self.net(spectrogram.transpose(1, 2))
        log_probs = nn.functional.log_softmax(output_net, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": spectrogram_lengths}
