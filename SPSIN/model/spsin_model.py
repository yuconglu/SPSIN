import torch
import torch.nn as nn
import torch.nn.functional as F
from afnonet import Block as AFNOBlock
from .physics_layers import PIDC_Encoder
from .spsin_core import SPSIN_Solver


class SPSIN(nn.Module):
    def __init__(self, args, edge_index):
        super(SPSIN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.alpha = args.alpha
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        encoder_type = getattr(args, 'encoder_type', 'PIDC')
        
        if encoder_type == 'PIDC':
            self.encoder = PIDC_Encoder(args)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim * self.lag, args.hidden_dim),
                nn.ReLU()
            )
        
        self.afno_backbone = AFNOBlock(
            dim=args.hidden_dim,
            num_blocks=getattr(args, 'num_afno_blocks', 8)
        )
        
        self.solver = SPSIN_Solver(
            edge_index=edge_index,
            model_type='Full',
            num_nodes=args.num_nodes,
            latent_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            alpha=args.alpha,
            height=args.height,
            width=args.width,
            afno_ref=self.afno_backbone,
            n_iters=getattr(args, 'imex_iters', 2),
            use_second_order=getattr(args, 'use_second_order', False),
            use_pidc_corrector=getattr(args, 'use_pidc_corrector', True)
        )
        
        self.solver.set_horizon(args.horizon)
        
        self.dec = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(args.hidden_dim, args.output_dim)
        )

    def forward(self, X, targets, teacher_forcing_ratio=0.5, apply_r_drop=False):
        if hasattr(self.encoder, '__class__') and self.encoder.__class__.__name__ == 'PIDC_Encoder':
            X_encoded = self.encoder(X)
        else:
            X_reshaped = X.transpose(1, 2).reshape(X.shape[0], self.num_node, self.input_dim * self.lag)
            X_encoded = self.encoder(X_reshaped)

        solver_out = self.solver(X_encoded)
        solver_out = solver_out.permute(1, 0, 2, 3)

        solver_out_dropped = self.dropout(solver_out)
        out1 = self.dec(solver_out_dropped)

        if apply_r_drop:
            solver_out_dropped_2 = self.dropout(solver_out)
            out2 = self.dec(solver_out_dropped_2)
            return out1, out2
        else:
            return out1
