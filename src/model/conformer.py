from torch import nn
import torch
from einops import rearrange, repeat
class GLUActivation(nn.Module):#implemetation in 
    def __init__(self,dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
    def forward(self,inputs):
        outputs, gates=inputs.chunk(2, dim=self.dim)
        return outputs*gates.sigmoid()
class SwishActivation(nn.Module):#implemetation in 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, inputs):
        return inputs*inputs.sigmoid()
class Conv2dSubsampling(nn.Module):#implemetation in 
    def __init__(self,in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsampling=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )
    def forward(self,inputs, inputs_lengths):        
        outputs=self.subsampling(inputs.unsqueeze(1))
        outputs=rearrange(outputs, 'b c l d -> b l (d c)')#change maybe
        #outputs=outputs.mean(dim=1)
        output_lengths=(inputs_lengths>>2)-1
        return outputs, output_lengths
class FeedForwardNet(nn.Module):
    def __init__(self,model_dim, expansion_factor=4, ffn_dropout=0.1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential=nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim*expansion_factor),
            SwishActivation(),
            nn.Dropout(ffn_dropout),
            nn.Linear(model_dim*expansion_factor, model_dim),
            nn.Dropout(ffn_dropout)
        )
    def forward(self,x):        
        return self.sequential(x)
class Conv(nn.Module):
    def __init__(self,model_dim, expansion_factor,p_dropout, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm=nn.LayerNorm(model_dim)
        self.sequential=nn.Sequential(
            nn.Conv1d(model_dim, model_dim*expansion_factor, kernel_size=1, stride=1, padding=0, bias=True),
            GLUActivation(dim=1),
            nn.Conv1d(
                model_dim, 
                model_dim, 
                kernel_size, 
                stride=1, 
                padding=(kernel_size - 1) // 2,
                groups=model_dim
            ),
            nn.BatchNorm1d(model_dim),
            SwishActivation(),
            nn.Conv1d(model_dim, model_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout(p_dropout)
        )
    def forward(self,input):        
        input_tr=self.layer_norm(input).transpose(1, 2)
        output=self.sequential(input_tr)
        return output.transpose(1, 2)
class MultiHeadSelfAttention(nn.Module):
    '''
    input: (B, L, D)
    '''
    def __init__(self,model_dim,num_heads, dim_head, p_dropout,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert model_dim%num_heads==0
        inner_dim=dim_head*num_heads
        self.heads=num_heads
        self.scale=dim_head**(-0.5)
        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(model_dim, 3*inner_dim, bias=False)
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(p_dropout)
        )
        self.layer_norm=nn.LayerNorm(model_dim)

    def forward(self,x):
        x=self.layer_norm(x)
        qkv=self.to_qkv(x).chunk(3, dim=-1)
        q, k, v=map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), qkv)
        dots=torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn=self.attend(dots)
        out=torch.matmul(attn, v)
        out=rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)
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
            attention_dropout=0.1,
            *args, 
            **kwargs
            ):
        super().__init__(*args, **kwargs)
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
            is_subsample=True,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.is_subsample=is_subsample
        self.spec_aug=None #Change
        self.conv_subsampling=Conv2dSubsampling(1, enc_dim)
        if(is_subsample):
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
        if(self.is_subsample):
            subsampled_specs, spectogram_lengths=self.conv_subsampling(spectorgram.transpose(1, 2), spectogram_lengths)
        else:
            subsampled_specs, spectogram_lengths=spectorgram.transpose(1, 2), spectogram_lengths
        before_conformers=self.dropout(self.linear1(subsampled_specs))
        after_conformers=self.conformer_blocks(before_conformers)
        log_probs=nn.functional.log_softmax(self.mlp(after_conformers), dim=-1)
        return {"log_probs":log_probs, "log_probs_length":spectogram_lengths}
        