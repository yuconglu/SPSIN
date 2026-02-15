import torch
import torch.nn as nn
import torch.nn.functional as F
from afnonet import Block as AFNOBlock


class PhysicalOperator(nn.Module):
    def __init__(self, num_nodes, height, width, latent_dim, embed_dim, alpha, model_type='Full'):
        super().__init__()
        self.num_nodes = num_nodes
        self.H = height
        self.W = width
        self.model_type = model_type
        self.alpha = alpha
        
        static_adj = self._build_moore_adjacency(height, width)
        self.register_buffer('static_adj', static_adj)
        
        self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.A2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
    
    def _build_moore_adjacency(self, height, width):
        N = height * width
        A = torch.zeros(N, N)
        
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_idx = ni * width + nj
                            A[idx, neighbor_idx] = 1.0
        
        A = A + torch.eye(N)
        D_inv = torch.diag(1.0 / (A.sum(1) + 1e-7))
        A_normalized = torch.matmul(D_inv, A)
        
        return A_normalized
    
    def forward(self, x):
        if self.model_type == 'diff':
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A1.T)))
        elif self.model_type == 'adv':
            A_out = F.relu(torch.tanh(self.alpha * (torch.mm(self.A1, self.A2.T) - torch.mm(self.A2, self.A1.T))))
        else:
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A2.T)))
        
        A_final = A_out * self.static_adj
        
        D_out = torch.diag(A_final.sum(1))
        L = D_out - A_final.T
        
        physical_term = -torch.matmul(L, x)
        
        return physical_term


class SpatialConfidenceGate(nn.Module):
    def __init__(self, height, width, in_channels):
        super().__init__()
        self.H = height
        self.W = width
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        x_grid = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        gate_grid = self.net(x_grid)
        
        gate = gate_grid.permute(0, 2, 3, 1).reshape(B, N, 1)
        
        return gate


class ExplicitPredictor(nn.Module):
    def __init__(self, graph_laplacian, afno_ref, spatial_gate, height, width):
        super().__init__()
        self.graph_phys = graph_laplacian
        self.afno = afno_ref
        self.spatial_gate = spatial_gate
        self.H = height
        self.W = width
    
    def forward(self, x):
        B, N, C = x.shape
        
        phys_term = self.graph_phys(x)
        
        x_grid = x.view(B, self.H, self.W, C)
        afno_out_grid = self.afno(x_grid)
        afno_term = afno_out_grid.reshape(B, N, C)
        
        gate = self.spatial_gate(x)
        
        output = phys_term + gate * afno_term
        
        return output


class SPSIN_Solver(nn.Module):
    def __init__(self, edge_index, model_type, num_nodes, latent_dim, embed_dim, 
                 alpha, height, width, afno_ref, n_iters=2, use_second_order=False, 
                 use_pidc_corrector=True):
        super().__init__()
        self.horizon = None
        self.H = height
        self.W = width
        self.n_iters = n_iters
        self.use_second_order = use_second_order
        self.use_pidc_corrector = use_pidc_corrector
        
        self.graph_laplacian = PhysicalOperator(
            num_nodes, height, width, latent_dim, embed_dim, alpha, model_type
        )
        
        self.spatial_gate = SpatialConfidenceGate(height, width, latent_dim)
        
        self.explicit_predictor = ExplicitPredictor(
            self.graph_laplacian, afno_ref, self.spatial_gate, height, width
        )
        
        self.corrector = self._build_corrector(latent_dim, use_pidc=self.use_pidc_corrector)
        
        if self.use_second_order:
            self.raw_damping_gamma = nn.Parameter(torch.tensor(-2.3))
            self.raw_correction_gamma = nn.Parameter(torch.tensor(0.0))
        else:
            self.raw_correction_gamma = nn.Parameter(torch.tensor(0.0))
    
    def _build_corrector(self, in_channels, hidden_dim=32, use_pidc=True):
        if use_pidc:
            from .physics_layers import PIDC_Conv2d
            
            class ImplicitCorrectorPIDC(nn.Module):
                def __init__(self, in_channels, hidden_dim):
                    super().__init__()
                    self.layer1 = PIDC_Conv2d(in_channels, hidden_dim, kernel_type='laplacian')
                    self.act = nn.Tanh()
                    self.layer2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
                    
                    with torch.no_grad():
                        nn.init.zeros_(self.layer2.weight)
                        nn.init.zeros_(self.layer2.bias)
                
                def forward(self, x_diff):
                    x = self.layer1(x_diff)
                    x = self.act(x)
                    x = self.layer2(x)
                    return x
            
            return ImplicitCorrectorPIDC(in_channels, hidden_dim)
        
        else:
            class ImplicitCorrectorCNN(nn.Module):
                def __init__(self, in_channels, hidden_dim):
                    super().__init__()
                    self.layer1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
                    self.act = nn.Tanh()
                    self.layer2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
                    
                    with torch.no_grad():
                        nn.init.zeros_(self.layer2.weight)
                        nn.init.zeros_(self.layer2.bias)
                
                def forward(self, x_diff):
                    x = self.layer1(x_diff)
                    x = self.act(x)
                    x = self.layer2(x)
                    return x
            
            return ImplicitCorrectorCNN(in_channels, hidden_dim)
    
    def set_horizon(self, horizon):
        self.horizon = horizon
    
    def _imex_correction(self, pred_x, current_x, B, N, C, correction_gamma):
        corrected_x = pred_x
        
        for k in range(self.n_iters):
            diff = corrected_x - current_x
            
            diff_grid = diff.view(B, self.H, self.W, C).permute(0, 3, 1, 2)
            delta_grid = self.corrector(diff_grid)
            
            delta = delta_grid.permute(0, 2, 3, 1).reshape(B, N, C)
            
            corrected_x = pred_x + correction_gamma * delta
        
        return corrected_x
    
    def forward(self, x, eval_times=None):
        if self.horizon is None:
            raise ValueError("Horizon not set. Call set_horizon() first.")
        
        B, N, C = x.shape
        
        correction_gamma = torch.sigmoid(self.raw_correction_gamma) * 0.2
        
        trajectory = []
        current_x = x
        
        if self.use_second_order:
            damping_gamma = F.softplus(self.raw_damping_gamma)
            
            m = torch.zeros_like(x).to(x.device)
            
            for step in range(self.horizon):
                force = self.explicit_predictor(current_x)
                
                m = m + (force - damping_gamma * m) * 1.0
                
                pred_x = current_x + m * 1.0
                
                corrected_x = self._imex_correction(pred_x, current_x, B, N, C, correction_gamma)
                
                trajectory.append(corrected_x)
                current_x = corrected_x
        
        else:
            for step in range(self.horizon):
                derivative = self.explicit_predictor(current_x)
                pred_x = current_x + derivative * 1.0
                
                corrected_x = self._imex_correction(pred_x, current_x, B, N, C, correction_gamma)
                
                trajectory.append(corrected_x)
                current_x = corrected_x
        
        return torch.stack(trajectory, dim=0)
