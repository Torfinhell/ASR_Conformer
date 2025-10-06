from torch import nn
class GLUActivation(nn.Module):
    def __init__(self,dim ):
        super().__init__()
        self.dim=dim
    def forward(self,inputs):
        outputs, gates=inputs.chunk(2, dim=self.dim)
        return outputs*gates.sigmoid()
class SwishActivation(nn.Module):
    def __init__(self ):
        super().__init__()
    def forward(self, inputs):
        return inputs*inputs.sigmoid()
class FeedForwardNet(nn.Module):
    def __init__(self,model_dim, expansion_factor=4, ffn_dropout=0.1,):
        super().__init__()
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