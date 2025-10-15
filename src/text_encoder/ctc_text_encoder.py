from string import ascii_lowercase
from typing import Iterable

from .base_text_encoder import BaseTextEncoder


class CTCTextEncoder(BaseTextEncoder):
    """CTC text encoder."""

    def __init__(self, alphabet: Iterable[str] | None = None, **kwargs) -> None:
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        super().__init__(alphabet, **kwargs)

    def get_splits(self, text: str) -> Iterable[str]:
        return [token for token in text]
