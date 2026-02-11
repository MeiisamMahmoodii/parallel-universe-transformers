"""Dragonnet Baseline.

Shi et al. (2019): Adapting Neural Networks for the Estimation of Treatment Effects.
Uses a shared representation Z(X) for propensity and outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class DragonnetModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Shared representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        # Propensity head
        self.propensity_head = nn.Linear(hidden_dim, 1)
        
        # Outcome heads (T=0 and T=1)
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        z = self.feature_extractor(x)
        
        prop_logits = self.propensity_head(z)
        y0 = self.head0(z)
        y1 = self.head1(z)
        
        return y0, y1, prop_logits


class DragonnetBaseline:
    """Wrapper for Dragonnet training and inference on the fly."""
    
    def __init__(self, device: str = "cpu", epochs: int = 50, lr: float = 1e-3, reg_alpha: float = 1.0):
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.alpha = reg_alpha # Weight for propensity loss (targeted reg)

    def __call__(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device
        
        predictions = torch.zeros(B, W, Nq, device=device)
        
        for b in range(B):
            X_sup = support_x[b, :, :-1]
            T_sup = support_x[b, :, -1].float()
            Y_sup = support_y[b]
            
            if support_mask is not None:
                m = support_mask[b].flatten().bool()
                if m.sum() < 4:
                    predictions[b] = Y_sup.mean() if len(Y_sup) > 0 else 0
                    continue
                X_sup = X_sup[m]
                T_sup = T_sup[m]
                Y_sup = Y_sup[m]
            
            T_bin = (T_sup >= 0.5).float()
            
            model = DragonnetModel(input_dim=d-1, hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                y0_pred, y1_pred, prop_logits = model(X_sup)
                
                # Outcome Loss
                # mask for T=0 and T=1
                # If T=0, loss on y0; if T=1, loss on y1
                y_pred = torch.where(T_bin.unsqueeze(1) == 0, y0_pred, y1_pred).squeeze()
                loss_y = F.mse_loss(y_pred, Y_sup)
                
                # Propensity Loss
                loss_prop = F.binary_cross_entropy_with_logits(prop_logits.squeeze(), T_bin)
                
                # Targeted Reg (optional, simple version just adds losses)
                total_loss = loss_y + self.alpha * loss_prop
                
                total_loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                for w in range(W):
                    X_q = query_x[b, w, :, :-1]
                    y0, y1, _ = model(X_q)
                    
                    if w == 0:
                        predictions[b, w] = y0.squeeze()
                    else:
                        predictions[b, w] = y1.squeeze()

        return {"prediction": predictions, "log_var": torch.zeros_like(predictions)}
