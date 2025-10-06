from torch import nn
import torch
from einops import rearrange, repeat
from math import log
class RelativePosEmb(nn.Module):
    '''
    input: (B, L, D)
    '''
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim=model_dim

    def forward(self,x): # (B, L, D)
        relative_ind=torch.arange(0, x.shape[1], dtype=torch.float32).unsqueeze(1)-torch.arange(0, x.shape[2], dtype=torch.float32).unsqueeze(0)
        relative_ind*=torch.exp(-torch.arange(0, 2*x.shape[2], 2, dtype=torch.float32)*(log(10000.0)/self.model_dim)).unsqueeze(0)
        relative_ind[:, ::2]=torch.cos(relative_ind[:, ::2])
        relative_ind[:, 1::2]=torch.sin(relative_ind[:, 1::2])
        return relative_ind.repeat(x.shape[0], 1, 1).to(x.device)
class MultiHeadSelfAttention(nn.Module):
    '''
    input: (B, L, D)
    '''
    def __init__(self,model_dim,num_heads, dim_head, p_dropout  ):
        super().__init__()
        assert model_dim%num_heads==0
        inner_dim=dim_head*num_heads
        self.heads=num_heads
        self.layer_norm=nn.LayerNorm(model_dim)
        self.pos_embedding=RelativePosEmb(model_dim)
        self.u_bias=nn.Parameter(torch.Tensor(num_heads, dim_head))
        self.v_bias=nn.Parameter(torch.Tensor(num_heads, dim_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.scale=dim_head**(-0.5)
        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(model_dim, 3*inner_dim)
        self.proj_emb=nn.Linear(model_dim, inner_dim, bias=False)
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(p_dropout)
        )

    def forward(self,x):
        x=self.layer_norm(x)
        pos_embedding=self.pos_embedding(x)
        qkv=self.to_qkv(x).chunk(3, dim=-1)
        pos_embedding=self.proj_emb(pos_embedding)
        q, k, v, pos_emb=map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), qkv+(pos_embedding,))
        content_score=torch.matmul(q+self.u_bias[None, :, None, :], k.transpose(-1, -2))
        pos_score = torch.matmul(q+self.v_bias[None, :, None, :], pos_emb.transpose(-1, -2))
        score=(pos_score+content_score)*self.scale
        attn=self.attend(score)
        out=torch.matmul(attn, v)
        out=rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)