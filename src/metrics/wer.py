from typing import List

import torch
from torch import Tensor

import editdistance
from collections import defaultdict
import numpy as np
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
        for ind_vec, length, target_text in zip(predictions, lengths, text):
            target_text=self.text_encoder.normalize(target_text)
            pred_text=self.text_encoder.ctc_decode(ind_vec[:length])
            wers.append(calc_wer(target_text, pred_text))   
        return sum(wers)/len(wers)
class BeamSearchWERMetric():
    def __init__(self, text_encoder, beam_size, beam_depth=1, take_first_chars=28):
        self.name="BEAM_WER"
        self.text_encoder=text_encoder
        self.beam_size=beam_size
        self.beam_depth=beam_depth
        self.take_first_chars=take_first_chars
    def __call__(self, probs, log_probs_length, text, **batch):
        wers=[]
        predictions=probs.cpu().numpy()
        lengths=log_probs_length.cpu().detach().numpy()
        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text=self.text_encoder.normalize(target_text)
            dp=self.ctc_beam_search(prob_vec, length)
            beam_pred = list(self.truncate_beams(dp, 1).keys())[0][0]
            wers.append(calc_wer(target_text, beam_pred))   
        return sum(wers)/len(wers)
    def ctc_beam_search(self,probs, length):
        dp={
            ("",self.text_encoder.EMPTY_TOK):1.0,
        }
        for idx in range(0,len(probs),self.beam_depth):
            for step in range(min(self.beam_depth, len(probs)-idx)):
                dp=self.expand_beam_and_merge_beams(dp,probs[idx+step], length)
            dp=self.truncate_beams(dp)
        return dp
    def expand_beam_and_merge_beams(self,dp, cur_step_prob, maxlen):
        new_dp=defaultdict(float)
        best_chars_sorted=np.argsort(cur_step_prob)[::-1]
        for (pref, prev_char), prev_proba in dp.items():
            for idx in best_chars_sorted[:self.take_first_chars]:
                char=self.text_encoder.ind2char[idx]
                cur_proba=prev_proba*cur_step_prob[idx]
                cur_pref=pref
                if char!=self.text_encoder.EMPTY_TOK and prev_char!=char:
                    cur_pref+=char
                if len(cur_pref)<=maxlen:
                    new_dp[(cur_pref, char)]+=cur_proba
        return new_dp

    def truncate_beams(self, dp, beam_size=None):
        beam_size = beam_size if beam_size else self.beam_size
        return dict(sorted(dp.items(), key=lambda x: -x[1])[:beam_size])