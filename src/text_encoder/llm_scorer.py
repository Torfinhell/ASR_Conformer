from typing import Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
import numpy as np
import torch
class LLMToScore:
    """LLM-based rescoring interface for beam_search prediction"""

    def __init__(self, device: str = 'cpu'):
        self.device=device
        self.model=GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer=GPT2Tokenizer.from_pretrained('gpt2')


    def score(self, text):
        token_tensor=self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            loss=self.model(token_tensor, labels=token_tensor)[0]
        return np.exp(loss.cpu().detach().numpy()).item()
