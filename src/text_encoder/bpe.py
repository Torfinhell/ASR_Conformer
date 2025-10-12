from collections import defaultdict
from string import ascii_lowercase

import torch

from .ctc_text_encoder import CTCTextEncoder


class BpeTrainer:
    """
    Text is already normalized
    """

    def __init__(self, vocab_size=100, min_alphabet=None):
        self.vocab = set(min_alphabet)
        self.vocab_size = vocab_size
        assert len(self.vocab) <= self.vocab_size

    def train(self, texts):
        self.word_freqs = defaultdict(int)
        self.merges = {}
        for text in texts:
            for word in text.split():
                self.word_freqs[word] += 1
        self.splits = {word: [c for c in word] for word in self.word_freqs}
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs()
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            self.update_split(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.add(best_pair[0] + best_pair[1])

    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freq
        return pair_freqs

    def update_split(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            ind = 0
            while ind < len(split) - 1:
                if split[ind] == a and split[ind + 1] == b:
                    split = split[:ind] + [a + b] + split[(ind + 2) :]
                else:
                    ind += 1
            self.splits[word] = split

    def encode(self, text):
        splits = [[chr for chr in word] for word in text.split()]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[(i + 2) :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])


class BpeEncoder(CTCTextEncoder):
    def __init__(self, vocab_size=100, texts=None, **kwargs):
        self.bpe_encoder = BpeTrainer(
            vocab_size, min_alphabet=list(ascii_lowercase + " ")
        )
        texts = [self.normalize_text(text) for text in texts]
        self.bpe_encoder.train(texts)
        super().__init__(list(self.bpe_encoder.vocab))

    def get_splits(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return self.bpe_encoder.encode(text)
