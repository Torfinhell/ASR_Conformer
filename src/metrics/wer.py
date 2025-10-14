from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric:
    def __init__(self, text_encoder, name):
        self.name = name
        self.text_encoder = text_encoder

    def __call__(self, log_probs, log_probs_length, text, **batch):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for ind_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(ind_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric:
    def __init__(
        self, text_encoder, name
    ):
        self.name = name
        self.text_encoder = text_encoder

    def __call__(self, probs, log_probs_length, text, **batch):
        wers = []
        predictions = probs.cpu().numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            dp = self.text_encoder.ctc_beam_search(prob_vec, length)
            if len(self.text_encoder.truncate_beams(dp, 1).keys()):
                beam_pred = list(self.text_encoder.truncate_beams(dp, 1).keys())[0][0]
            else:
                beam_pred = ""
            wers.append(calc_wer(target_text, beam_pred))
        return sum(wers) / len(wers)