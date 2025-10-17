from typing import Any, Dict

import torch
from torch import nn

from .activation_and_ffn import FeedForwardNet
from .attention import MultiHeadSelfAttention
from .conv import Conv, Conv2dSubsampling


class ConformerBlock(nn.Module):
    """Conformer block composing FFN, MHSA and ConvolutionBlock.

    Input:
    - x: Tensor with shape (B, L, D)

    Returns
    - Tensor with shape (B, L, D)
    """

    def __init__(
        self,
        model_dim: int,
        ffn_expansion_factor: int = 4,
        ffn_dropout: float = 0.1,
        num_attention_heads: int = 4,
        dim_head: int = 64,
        conv_dropout: float = 0.1,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardNet(model_dim, ffn_expansion_factor, ffn_dropout)
        self.mhsa = MultiHeadSelfAttention(
            model_dim, num_attention_heads, dim_head, attention_dropout
        )
        self.conv = Conv(
            model_dim, conv_expansion_factor, conv_dropout, conv_kernel_size
        )
        self.ffn2 = FeedForwardNet(model_dim, ffn_expansion_factor, ffn_dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn1(x) / 2
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        return self.layer_norm(x + self.ffn2(x) / 2)


class Conformer(nn.Module):
    """Conformer model producing probabilities(and log_probs also) for tokens.

    Input:
    - spectrogram: Tensor with shape (B, F, T)
    - spectrogram_lengths: Tensor with shape (B,)

    Returns
    - dict with keys: log_probs, probs, log_probs_length
    """

    def __init__(
        self,
        input_dim: int,
        n_tokens: int,
        num_attention_heads: int,
        dim_head: int,
        model_dim: int,
        num_conformer_blocks: int,
        conv_kernel_size: int = 31,
        input_dropout: float = 0.1,
        conv_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        do_downsample: bool = True,
        unfreeze_last_layers:int=None
    ) -> None:
        super().__init__()
        self.do_downsample = do_downsample
        self.conv_subsampling = Conv2dSubsampling(1, 1)
        self.linear1 = nn.Linear((input_dim // 4 - 1), model_dim)
        self.dropout = nn.Dropout(input_dropout)
        self.conformer_blocks = nn.Sequential(
            *[
                ConformerBlock(
                    model_dim=model_dim,
                    num_attention_heads=num_attention_heads,
                    conv_dropout=conv_dropout,
                    ffn_dropout=ffn_dropout,
                    ffn_expansion_factor=ffn_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attention_dropout=attention_dropout,
                    dim_head=dim_head,
                )
                for _ in range(num_conformer_blocks)
            ]
        )
        assert unfreeze_last_layers is None or unfreeze_last_layers<=num_conformer_blocks
        if unfreeze_last_layers is not None:
            for conformer_block in self.conformer_blocks[:-unfreeze_last_layers]:
                for param in conformer_block.parameters():
                    param.requires_grad=False
        self.mlp = nn.Linear(model_dim, n_tokens, bias=False)

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_lengths: torch.Tensor, **batch: Any
    ) -> Dict[str, torch.Tensor]:
        if self.do_downsample:
            subsampled_specs, spectrogram_lengths = self.conv_subsampling(
                spectrogram.transpose(1, 2), spectrogram_lengths
            )
        else:
            subsampled_specs, spectrogram_lengths = (
                spectrogram.transpose(1, 2),
                spectrogram_lengths,
            )
        before_conformers = self.dropout(self.linear1(subsampled_specs))
        after_conformers = self.conformer_blocks(before_conformers)
        probs = nn.functional.softmax(self.mlp(after_conformers), dim=-1)
        log_probs = nn.functional.log_softmax(self.mlp(after_conformers), dim=-1)
        return {
            "log_probs": log_probs,
            "probs": probs,
            "log_probs_length": spectrogram_lengths,
        }
