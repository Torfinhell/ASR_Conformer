import re
from string import ascii_lowercase

import torch
# TODO add, LM support
import numpy as np
from collections import defaultdict
from .llm_scorer import LLMToScore
class BaseTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet,beam_size=None,beam_depth=None,take_first_chars=None, use_LLM=False,**kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.beam_size = beam_size
        self.beam_depth = beam_depth
        self.take_first_chars = take_first_chars
        try:
            self.beam_values_are_intialised()
            self.is_beam_search=True
        except(ValueError):
            self.is_beam_search=False
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2token = dict(enumerate(self.vocab))
        self.token2ind = {v: k for k, v in self.ind2token.items()}
        self.useLLM=use_LLM
        if use_LLM:
            self.llm=LLMToScore()

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2token[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(
                [self.token2ind[token] for token in self.get_splits(text)]
            ).unsqueeze(0)
        except KeyError:
            unknown_tokens = set(
                [token for token in self.get_splits(text) if token not in self.token2ind]
            )
            raise Exception(
                f"Can't encode text '{text}'. Unknown tokens: '{' '.join(unknown_tokens)}'"
            )

    def get_splits(self, text):
        raise NotImplementedError()

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2token[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        prev = None
        res = ""
        for ind in inds:
            if prev != ind and ind:
                res += self.ind2token[ind]
            prev = ind
        return res

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
    def ctc_beam_search(self, probs, length):
        self.beam_values_are_intialised()
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for idx in range(0, len(probs), self.beam_depth):
            for step in range(min(self.beam_depth, len(probs) - idx)):
                dp = self.expand_beam_and_merge_beams(dp, probs[idx + step], length)
            dp = self.truncate_beams(dp)
        return dp
    def expand_beam_and_merge_beams(self, dp, cur_step_prob, maxlen):
        self.beam_values_are_intialised()
        new_dp = defaultdict(float)
        best_chars_sorted = np.argsort(cur_step_prob)[::-1]
        for (pref, prev_char), prev_proba in dp.items():
            for idx in best_chars_sorted[: self.take_first_chars]:
                char = self.ind2token[idx]
                cur_proba = prev_proba * cur_step_prob[idx]
                cur_pref = pref
                if char != self.EMPTY_TOK and prev_char != char:
                    cur_pref += char
                if len(cur_pref) <= maxlen:
                    new_dp[(cur_pref, char)] += cur_proba
        return new_dp
    def truncate_beams(self, dp, beam_size=None):
        self.beam_values_are_intialised()
        beam_size = beam_size if beam_size else self.beam_size
        if(self.useLLM):
            return dict(sorted(dp.items(), key=lambda x: -self.llm.score(x[0]))[:beam_size])
        return dict(sorted(dp.items(), key=lambda x: -x[1])[:beam_size])
    def beam_values_are_intialised(self):
        if(self.beam_size is None or self.beam_depth is None or self.take_first_chars is None):
            raise ValueError("One of the beam_size, beam_depth or take_first_chars is not initialised in the config")