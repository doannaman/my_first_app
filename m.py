import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
import numpy as np
batch = 32
class Img_Embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=16,
                stride=16),
            torch.nn.Flatten(2)
        )
        self.cls_token = torch.nn.Parameter(torch.randn((1, 128, 1))) 
        self.row_embed = torch.nn.Embedding(20, 64) 
        self.col_embed = torch.nn.Embedding(20, 64)  
        self.reset_parameters()
        self.cls_pos = torch.nn.Parameter(torch.randn(128, 1)) 
    def reset_parameters(self):
        torch.nn.init.uniform_(self.row_embed.weight)
        torch.nn.init.uniform_(self.col_embed.weight)
    def forward(self, x):
        B = x.shape[0] 
        i_j = torch.arange(14, device=x.device)
        x_embed = self.row_embed(i_j).unsqueeze(1).repeat(1, 14, 1) 
        y_embed = self.col_embed(i_j).unsqueeze(0).repeat(14, 1, 1) 
        pos = torch.cat((x_embed, y_embed), dim=2).permute(2, 0, 1).flatten(1) 
        allpos = torch.cat((self.cls_pos, pos), dim=1) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x_patches = self.embed(x) 
        out = torch.cat((cls_tokens, x_patches), dim=2) 
        out = out + allpos.unsqueeze(0).expand(B, -1, -1) 
        return out.permute(2, 0, 1)
class Laynorm(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(128)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))
class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)
class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(128, 768),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 128),
            torch.nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        return self.dropout(attn_output)
class ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Img_Embed()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                Residual(Laynorm(Attention())),
                Residual(Laynorm(FeedForward()))
            ) for _ in range(10)
        ])
        self.cls = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.embed(x) 
        for layer in self.layers:
            x = layer(x)
        out_cls = x[0, :, :] 
        res = self.cls(out_cls) 
        return res