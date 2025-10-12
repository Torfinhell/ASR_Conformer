from collections import defaultdict
from string import ascii_lowercase

import torch
from ctc_text_encoder import CTCTextEncoder


class BpeTrainer:
    def __init__(self, vocab_size=100, min_alphabet=None):
        assert len(self.vocab <= self.vocab_size)
        self.vocab = set(min_alphabet)
        self.vocab_size = vocab_size

    def train(self, texts):
        while len(self.vocab) < self.vocab_size:
            encoded_texts = [self.encode(text) for text in texts]
            pair_freqs = defaultdict(int)
            for encoded_text in encoded_texts:
                for i in range(len(encoded_text) - 1):
                    pair_freqs[(encoded_text[i], encoded_text[i + 1])] += 1
            max_pair_freq = max(pair_freqs, key=pair_freqs.get)
            self.vocab.add(max_pair_freq[0] + max_pair_freq[1])

    def encode(self, text):
        indices = []
        iterate_text = text
        while iterate_text:
            encode_text = ""
            for char in iterate_text:
                if (encode_text + char) not in self.vocab:
                    indices += [self.vocab[encode_text]]
                    iterate_text = iterate_text[len(encode_text) :]
                    break
                encode_text += char
            if encode_text == iterate_text:
                raise Exception(
                    f"Can't encode text: '{text}'. Unknown to vocab string: '{encode_text}'"
                )
        return indices


class BpeEncoder(CTCTextEncoder):
    def __init__(self, vocab_size=100, texts=None, **kwargs):
        self.bpe_encoder = BpeTrainer(
            vocab_size, min_alphabet=list(ascii_lowercase + " ")
        )
        texts = [self.normalize_text(text) for text in texts]
        self.bpe_encoder.train(texts)
        super().__init__(list(self.bpe_encoder.vocab))

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.Tensor(self.bpe_encoder.encode(text)).unsqueeze(0)
