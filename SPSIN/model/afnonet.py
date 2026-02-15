import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.block_size = self.hidden_size // self.num_blocks
        
        self.w1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size))
    
    def forward(self, x):
        B, H, W, C = x.shape
        
        x = x.permute(0, 3, 1, 2)
        
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        x_ft = x_ft.reshape(B, self.num_blocks, self.block_size, H, W // 2 + 1)
        
        x_ft_real = x_ft.real
        x_ft_imag = x_ft.imag
        x_ft_stack = torch.stack([x_ft_real, x_ft_imag], dim=1)
        
        o1_real = F.relu(
            torch.einsum('bncxy,ncd->bndxy', x_ft_stack[0], self.w1[0]) + 
            self.b1[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        o1_imag = F.relu(
            torch.einsum('bncxy,ncd->bndxy', x_ft_stack[1], self.w1[1]) + 
            self.b1[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        
        o2_real = (
            torch.einsum('bndxy,ndc->bncxy', o1_real, self.w2[0]) + 
            self.b2[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        o2_imag = (
            torch.einsum('bndxy,ndc->bncxy', o1_imag, self.w2[1]) + 
            self.b2[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        
        x_ft = torch.complex(o2_real, o2_imag)
        
        x_ft = x_ft.reshape(B, C, H, W // 2 + 1)
        
        x = torch.fft.irfft2(x_ft, s=(H, W), norm='ortho')
        
        x = x.permute(0, 2, 3, 1)
        
        return x


class Block(nn.Module):
    def __init__(self, dim, num_blocks=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.afno = AFNO2D(dim, num_blocks=num_blocks)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.afno(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
