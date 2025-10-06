from torch import nn
from .attention import MultiHeadSelfAttention
from .conv import Conv2dSubsampling, Conv
from .activation_and_ffn import FeedForwardNet
import torch

class ConformerBlock(nn.Module):
    def __init__(
            self, 
            model_dim, 
            ffn_expansion_factor=4, 
            ffn_dropout=0.1,
            num_attention_heads=4,
            dim_head=64,
            conv_dropout=0.1,
            conv_expansion_factor=2,
            conv_kernel_size=32,
            attention_dropout=0.1
            ):
        super().__init__()
        self.ffn1=FeedForwardNet(model_dim, ffn_expansion_factor, ffn_dropout)
        self.mhsa=MultiHeadSelfAttention(model_dim, num_attention_heads, dim_head, attention_dropout)
        self.conv=Conv(model_dim, conv_expansion_factor, conv_dropout, conv_kernel_size)
        self.ffn2=FeedForwardNet(model_dim, ffn_expansion_factor, ffn_dropout)
        self.layer_norm=nn.LayerNorm(model_dim)
    def forward(self, x):        
        x=x+self.ffn1(x)/2
        x=x+self.mhsa(x)
        x=x+self.conv(x)
        return self.layer_norm(x+self.ffn2(x)/2)
class Conformer(nn.Module):
    '''
    input: (B, L, D)
    conv_subsampling: (B, L//4-1, D)
    conv-block: (B, L//4-1, D)

    '''
    def __init__(
            self,
            model_dim,
            num_tokens,
            num_attention_heads,
            dim_head,
            enc_dim,
            num_conformer_blocks, 
            conv_kernel_size=32,
            input_dropout=0.1,
            conv_dropout=0.1,
            ffn_dropout=0.1,
            attention_dropout=0.1,
            ffn_expansion_factor=4,
            conv_expansion_factor=2,
            do_downsample=True,
        ):
        super().__init__()
        self.do_downsample=do_downsample
        self.spec_aug=None #Change
        self.conv_subsampling=Conv2dSubsampling(1, enc_dim)
        if(do_downsample):
            self.linear1=nn.Linear((model_dim//4-1)*enc_dim, model_dim)
        else:
            self.linear1=nn.Linear(model_dim, model_dim)
        self.dropout=nn.Dropout(input_dropout)
        self.conformer_blocks=nn.Sequential(*[
            ConformerBlock(
                model_dim=model_dim,
                num_attention_heads=num_attention_heads,
                conv_dropout=conv_dropout,
                ffn_dropout=ffn_dropout,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                attention_dropout=attention_dropout,
                dim_head=dim_head
            ) for _ in range(num_conformer_blocks)
        ])
        self.mlp=nn.Linear(model_dim, num_tokens, bias=False)
    def forward(self,spectorgram, spectogram_lengths, **batch):
        if(self.do_downsample):
            subsampled_specs, spectogram_lengths=self.conv_subsampling(spectorgram.transpose(1, 2), spectogram_lengths)
        else:
            subsampled_specs, spectogram_lengths=spectorgram.transpose(1, 2), spectogram_lengths
        before_conformers=self.dropout(self.linear1(subsampled_specs))
        after_conformers=self.conformer_blocks(before_conformers)
        probs=nn.functional.softmax(self.mlp(after_conformers), dim=-1)
        log_probs = nn.functional.log_softmax(self.mlp(after_conformers), dim=-1)
        return {"log_probs":log_probs, "probs":probs, "log_probs_length":spectogram_lengths}
        