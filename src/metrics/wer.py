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

    def __init__(self, text_encoder, name: str, use_llm:bool=True) -> None:
        self.name = name
        self.text_encoder = text_encoder
        self.use_llm=use_llm
    def __call__(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text: Iterable[str],
        **batch,
    ) -> float:
        wers = []
        predictions = log_probs.cpu().numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            dp = self.text_encoder.ctc_beam_search(log_prob_vec, length)
            beams = self.text_encoder.truncate_beams(dp, beam_size=1, use_llm=self.use_llm)
            if beams:
                beam_pred = list(beams.keys())[0][0]
            else:
                beam_pred = ""
            wers.append(calc_wer(target_text, beam_pred))
        return sum(wers) / len(wers)
