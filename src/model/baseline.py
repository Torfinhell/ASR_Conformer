from torch import nn


class BaseLineModel(nn.Module):
    def __init__(self, model_dim, n_tokens, fc_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            # input Size:(N, in_feauterse, time_len)
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
