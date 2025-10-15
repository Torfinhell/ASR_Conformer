from typing import Dict

import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(zero_infinity=True,*args, **kwargs)
    def forward(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text_encoded: Tensor,
        text_encoded_lengths: Tensor,
        **batch,
    ) -> Dict[str, Tensor]:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_lengths,
        )

        return {"loss": loss}
