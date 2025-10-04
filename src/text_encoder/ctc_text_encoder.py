import re
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder():
    EMPTY_TOK=""
    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet=list(ascii_lowercase+" ")
        self.alphabet=alphabet
        self.vocab=[self.EMPTY_TOK]+list(self.alphabet)
        self.ind2char=dict(enumerate(self.vocab))
        self.char2ind={v:k for k, v in self.ind2char.items()}
    def __len__(self):
        return len(self.vocab)
    def __getitem__(self, item:int):
        assert type(item) is int
        return self.ind2char[item]
    def encode(self, text)->torch.Tensor:
        text=self.normalize(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars=set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Cant encode text: {text}. Unknown Chars:{' '.join(unknown_chars)}"
            )
    def decode(self, inds)->str:
        return ''.join([self.ind2char[ind] for ind in inds ]).strip()
    def ctc_decode(self, inds) -> str: #change
        prev=None
        res=""
        for ind in inds:
            if(prev!=ind and ind):  
                res+=self.ind2char[ind]
            prev=ind
        return res
    @staticmethod
    def normalize(text):
        text=text.lower()
        text=re.sub(r"[^a-z ]", "", text)
        return text