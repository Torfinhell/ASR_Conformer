from typing import Any


class LLMToScore:
    """LLM-based rescoring interface for beam_search prediction"""

    def __init__(self):
        pass

    def score(self, text):
        raise NotImplementedError()
