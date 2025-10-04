from typing import List

import torch
from torch import Tensor

import editdistance
def calc_wer(target_text:str, pred_text:str):
    assert len(target_text)
    return editdistance.eval(target_text.split(), pred_text.split())/len(target_text.split())

class ArgmaxWERMetric():
    def __init__(self, text_encoder):
        self.name="ARGMAX_WER"
        self.text_encoder=text_encoder
    def __call__(self, log_probs, log_probs_length, text, **batch):
        wers=[]
        predictions=torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths=log_probs_length.cpu().detach().numpy()
        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text=self.text_encoder.normalize(target_text)
            pred_text=self.text_encoder.ctc_decode(prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))   
        return sum(wers)/len(wers)