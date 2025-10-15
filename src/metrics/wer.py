from typing import Iterable

import torch

from src.metrics.utils import calc_wer


class ArgmaxWERMetric:
    """Compute WER by greedy (argmax) decoding.

    Args:
        text_encoder: has helper functions for processing text and decoding
        name: metric name.
    Returns
    - float: mean WER over the batch
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
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for ind_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(ind_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric:
    """Compute WER using CTC beam search decoding.

    Args:
        text_encoder: has helper functions for processing text and decoding beam_search
        name: metric name.

    Returns
    - float: mean WER over the batch
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
        wers = []
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
            wers.append(calc_wer(target_text, beam_pred))
        return sum(wers) / len(wers)
