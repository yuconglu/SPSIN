import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDC_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_type='laplacian'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_type = kernel_type
        
        if kernel_type == 'laplacian':
            fixed_kernel = torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ]).unsqueeze(0).unsqueeze(0)
        else:
            fixed_kernel = torch.tensor([
                [1., 1., 1.],
                [1., -8., 1.],
                [1., 1., 1.]
            ]).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('fixed_kernel', fixed_kernel)
        
        self.learnable_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        phys_output_list = []
        for c in range(C):
            x_c = x[:, c:c+1, :, :]
            phys_c = F.conv2d(x_c, self.fixed_kernel, padding=1)
            phys_output_list.append(phys_c)
        phys_output = torch.cat(phys_output_list, dim=1)
        
        learnable_output = self.learnable_conv(x)
        
        alpha_sig = torch.sigmoid(self.alpha)
        output = alpha_sig * phys_output + (1 - alpha_sig) * learnable_output
        
        return output


class PIDC_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.lag = args.lag
        self.H = args.height
        self.W = args.width
        
        self.conv1 = PIDC_Conv2d(
            in_channels=self.input_dim * self.lag,
            out_channels=args.hidden_dim // 2,
            kernel_type='laplacian'
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=args.hidden_dim // 2,
            out_channels=args.hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        self.act = nn.ReLU()
    
    def forward(self, X):
        B, T, N, D = X.shape
        
        X_reshaped = X.transpose(1, 2).reshape(B, N, T * D)
        
        X_grid = X_reshaped.view(B, self.H, self.W, T * D).permute(0, 3, 1, 2)
        
        h1 = self.act(self.conv1(X_grid))
        h2 = self.act(self.conv2(h1))
        
        X_encoded = h2.permute(0, 2, 3, 1).reshape(B, N, self.args.hidden_dim)
        
        return X_encoded
