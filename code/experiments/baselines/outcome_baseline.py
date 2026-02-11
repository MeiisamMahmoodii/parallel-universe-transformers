"""Outcome-model baseline for comparison protocol.

Fits a simple ridge regression on the support set (per batch item) and predicts
E[Y|X] at each world's query covariates. Deltas are (prediction for world w - prediction for world 0).
Implements the same interface as the model: (support_x, support_y, query_x, ...) -> prediction, log_var.
Useful as a non-trivial baseline on synthetic SCM data; no causal graph.
"""

import torch
import numpy as np
from typing import Dict, Optional


class OutcomeBaseline:
    """Predict using ridge regression fit on support (x,y); predict at query x per world. Deltas = world w - baseline."""

    def __init__(self, device: str = "cpu", ridge: float = 1.0):
        self.device = torch.device(device)
        self.ridge = ridge

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
        """Same signature as ParallelUniverseTransformer.forward; returns prediction and log_var."""
        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device

        support_x_np = support_x.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_x_np = query_x.detach().cpu().numpy()

        predictions = np.zeros((B, W, Nq), dtype=np.float32)
        for b in range(B):
            X_sup = support_x_np[b]   # [Ns, d]
            Y_sup = support_y_np[b]   # [Ns]
            if support_mask is not None:
                m = support_mask[b].detach().cpu().numpy()
                valid = m.any(axis=1) # Or flatten if 1D mask
                # Check shape of mask
                if m.ndim > 1:
                     valid = m.any(axis=1)
                else:
                     valid = m.astype(bool)
                     
                if valid.sum() < 2:
                    Y_sup = np.broadcast_to(np.nanmean(Y_sup), (Ns,))
                else:
                    X_sup = X_sup[valid]
                    Y_sup = Y_sup[valid]
            
            # Simple Ridge: beta = (X'X + lambda I)^-1 X'y
            # Add bias
            X_sup = np.hstack([X_sup, np.ones((X_sup.shape[0], 1))])
            try:
                beta = np.linalg.solve(
                    X_sup.T @ X_sup + self.ridge * np.eye(X_sup.shape[1]),
                    X_sup.T @ Y_sup,
                )
            except np.linalg.LinAlgError:
                beta = np.zeros(X_sup.shape[1])
                beta[-1] = np.nanmean(Y_sup)
                
            for w in range(W):
                X_q = query_x_np[b, w, :, :]  # [Nq, d]
                X_q = np.hstack([X_q, np.ones((X_q.shape[0], 1))])
                predictions[b, w, :] = X_q @ beta

        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
