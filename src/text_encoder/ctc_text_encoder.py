from .base_text_encoder  import BaseTextEncoder
from string import ascii_lowercase
class CTCTextEncoder(BaseTextEncoder):
    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        super().__init__(alphabet)
    def get_splits(self, text):
        return [token for token in text]