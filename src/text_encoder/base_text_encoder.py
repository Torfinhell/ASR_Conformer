import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Iterable, Optional, Any

import numpy as np
import torch
from torch import Tensor

from .llm_scorer import LLMToScore
class BaseTextEncoder:
    """Base class for text encoders.

    Args:
        alphabet: list of symbols
        beam_size, beam_depth, take_first_tokens: parametrs for beam-search
    """

    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet: Optional[Iterable[str]] = None,
        beam_size: Optional[int] = None,
        beam_depth: Optional[int] = None,
        take_first_tokens: Optional[int] = None,
        llm_model: Any = None,
        **kwargs,
    ) -> None:
        self.beam_size = beam_size
        self.beam_depth = beam_depth
        self.take_first_tokens = take_first_tokens
        try:
            self.beam_values_are_intialised()
            self.is_beam_search = True
        except ValueError:
            self.is_beam_search = False
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2token = dict(enumerate(self.vocab))
        self.token2ind = {v: k for k, v in self.ind2token.items()}
        self.llm = llm_model
    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, item: int) -> str:
        assert type(item) is int
        return self.ind2token[item]

    def encode(self, text: str) -> Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(
                [self.token2ind[token] for token in self.get_splits(text)]
            ).unsqueeze(0)
        except KeyError:
            unknown_tokens = set(
                [
                    token
                    for token in self.get_splits(text)
                    if token not in self.token2ind
                ]
            )
            raise Exception(
                f"Can't encode text '{text}'. Unknown tokens: '{' '.join(unknown_tokens)}'"
            )

    def get_splits(self, text: str) -> Iterable[str]:
        raise NotImplementedError()

    def decode(self, inds: Iterable[int]) -> str:
        return "".join([self.ind2token[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds: Iterable[int]) -> str:
        prev = None
        res = ""
        for ind in inds:
            if prev != ind and ind:
                res += self.ind2token[ind]
            prev = ind
        return res

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def ctc_beam_search(self, log_probs: np.ndarray, length: int):
        self.beam_values_are_intialised()
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for idx in range(0, len(log_probs), self.beam_depth):
            for step in range(min(self.beam_depth, len(log_probs) - idx)):
                dp = self.expand_beam_and_merge_beams(dp, log_probs[idx + step], length)
            dp = self.truncate_beams(dp)
        return dp

    def expand_beam_and_merge_beams(
        self, dp: dict, cur_step_log_prob: np.ndarray, maxlen: int
    ) -> dict:
        self.beam_values_are_intialised()
        new_dp = defaultdict(lambda: float('-inf'))
        best_tokens_sorted = np.argsort(cur_step_log_prob)[::-1]
        for (pref, prev_token), prev_log_proba in dp.items():
            for idx in best_tokens_sorted[:self.take_first_tokens]:
                token = self.ind2token[idx]
                cur_log_proba = prev_log_proba + cur_step_log_prob[idx]
                cur_pref = pref
                if token != self.EMPTY_TOK and prev_token != token:
                    cur_pref += token
                if len(cur_pref) <= maxlen:
                    new_dp[(cur_pref, token)]=np.logaddexp(new_dp[(cur_pref, token)],cur_log_proba)
        return new_dp

    def truncate_beams(self, dp: dict, beam_size: Optional[int] = None, use_llm:bool=False) -> dict:
        self.beam_values_are_intialised()
        beam_size = beam_size if beam_size else self.beam_size
        if use_llm and self.llm is not None:
            return dict(
                sorted(dp.items(), key=lambda x: self.llm.score(x[0]))[:beam_size]
            )
        return dict(sorted(dp.items(), key=lambda x: -x[1])[:beam_size])

    def beam_values_are_intialised(self):
        if (
            self.beam_size is None
            or self.beam_depth is None
            or self.take_first_tokens is None
        ):
            raise ValueError(
                "One of the beam_size, beam_depth or take_first_tokens is not initialised in the config"
            )
