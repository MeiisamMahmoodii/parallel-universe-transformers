"""TransTEE: Transformer-based Treatment Effect Estimator.

Simplified implementation based on Zhang et al. (2022).
Uses a Transformer to process covariates and treatments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class TransTeeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        self.treatment_embedding = nn.Embedding(2, hidden_dim) # Binary treatment assumption
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, t):
        # x: [B, d]
        # t: [B] (0 or 1)
        
        h_x = self.feature_embedding(x) # [B, H]
        h_t = self.treatment_embedding(t) # [B, H]
        
        # In TransTEE, they often concat or add. We'll use a sequence approach for Transformer.
        # Sequence: [x_emb, t_emb]
        
        seq = torch.stack([h_x, h_t], dim=1) # [B, 2, H]
        
        out_seq = self.transformer(seq)
        
        # Pool or take token0
        out = out_seq[:, 0, :] # Use x token context
        
        y0 = self.head0(out)
        y1 = self.head1(out)
        
        return y0, y1


class TransTEEBaseline:
    """Wrapper for TransTEE training and inference on the fly."""
    
    def __init__(self, device: str = "cpu", epochs: int = 50, lr: float = 1e-3):
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr

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
        
        # TransTEE needs training on support set.
        # Since this is "In-Context" evaluation, strictly we should train on support_x.
        # This might be slow for large benchmarks if we train per episode.
        # We will do a fast adaptation (few epochs).
        
        predictions = torch.zeros(B, W, Nq, device=device)
        
        for b in range(B):
            X_sup = support_x[b, :, :-1] # Covariates
            T_sup = support_x[b, :, -1].long().clamp(0, 1) # Treat (binarized)
            Y_sup = support_y[b]
            
            if support_mask is not None:
                m = support_mask[b].flatten().bool()
                if m.sum() < 2:
                    predictions[b] = Y_sup.mean() if len(Y_sup) > 0 else 0
                    continue
                X_sup = X_sup[m]
                T_sup = T_sup[m]
                Y_sup = Y_sup[m]
            
            # Init Model
            model = TransTeeModel(input_dim=d-1, hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                pred0, pred1 = model(X_sup, T_sup) # Note: TransTEE input usually just X? 
                # Re-reading: TransTEE usually takes X+T to predict Y?
                # Or is it a T-learner style? 
                # Implementation above predicts both Y(0) and Y(1) from X (and T in seq?).
                # Actually, standard T-Net predicts Y given X, T.
                # If we pass T, we should predict Y corresponding to T.
                
                # Let's adjust forward: simple T-learner logic inside NN
                # Gather correct head based on T
                
                pred_y = torch.where(T_sup.unsqueeze(1) == 0, pred0, pred1).squeeze()
                loss = criterion(pred_y, Y_sup)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                for w in range(W):
                    X_q = query_x[b, w, :, :-1]
                    # We want prediction for treatment = w
                    t_w = torch.full((Nq,), w, device=device, dtype=torch.long).clamp(0, 1)
                    
                    p0, p1 = model(X_q, t_w)
                    if w == 0:
                        predictions[b, w] = p0.squeeze()
                    else:
                        predictions[b, w] = p1.squeeze()

        return {"prediction": predictions, "log_var": torch.zeros_like(predictions)}
