from collections import defaultdict
from typing import Iterable, List

import numpy as np
import torch

from src.metrics.utils import calc_cer


class ArgmaxCERMetric:
    """Compute CER using greedy (argmax) decoding.

    Args:
        text_encoder: object implementing ``normalize_text`` and ``ctc_decode``.
        name: metric name.
    """

    def __init__(self, text_encoder, name: str) -> None:
        self.name = name
        self.text_encoder = text_encoder

    def __call__(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text: Iterable[str],
        **batch,
    ) -> float:
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for ind_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(ind_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric:
    """Compute CER using CTC beam search decoding.

    Args:
        text_encoder: Has helper function to decode beam_search
        name: metric name
    """

    def __init__(self, text_encoder, name: str) -> None:
        self.name = name
        self.text_encoder = text_encoder

    def __call__(
        self,
        probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text: Iterable[str],
        **batch,
    ) -> float:
        cers = []
        predictions = probs.cpu().numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            dp = self.text_encoder.ctc_beam_search(prob_vec, length)
            beams = self.text_encoder.truncate_beams(dp, 1)
            if len(beams.keys()):
                beam_pred = list(beams.keys())[0][0]
            else:
                beam_pred = ""
            cers.append(calc_cer(target_text, beam_pred))
        return sum(cers) / len(cers)
