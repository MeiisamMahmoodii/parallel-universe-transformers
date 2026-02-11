"""Linear Regression baselines (T-Learner and S-Learner).

T-Learner: Fits two separate linear models, one for treatment group and one for control.
S-Learner: Fits a single linear model with treatment as a feature.
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None


class LinearTBaseline:
    """T-Learner using Linear Regression (Ridge)."""

    def __init__(self, device: str = "cpu", alpha: float = 1.0):
        self.device = torch.device(device)
        self.alpha = alpha

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
        """Predicts using separate linear models for each treatment/world."""
        if Ridge is None:
            raise RuntimeError("sklearn is required for LinearBaseline")
            
        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device

        support_x_np = support_x.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_x_np = query_x.detach().cpu().numpy()

        # Last column is treatment/world indicator in support_x
        X_cov_sup = support_x_np[:, :, :-1]
        t_sup = support_x_np[:, :, -1]

        predictions = np.zeros((B, W, Nq), dtype=np.float32)
        
        for b in range(B):
            X_cov = X_cov_sup[b]
            Y = support_y_np[b]
            t = t_sup[b]
            
            if support_mask is not None:
                m = support_mask[b].detach().cpu().numpy().flatten().astype(bool)
                if m.sum() < 2:
                    predictions[b, :, :] = np.nanmean(Y) if len(Y) > 0 else 0.0
                    continue
                X_cov = X_cov[m]
                Y = Y[m]
                t = t[m]

            # Fit separate models for each world w (assuming w corresponds to t value)
            # Usually w=0 is control, w=1 is treated. 
            # If W > 2, this assumes t takes values 0..W-1.
            
            models = {}
            # Group data by treatment value
            unique_t = np.unique(t)
            
            global_mean = np.nanmean(Y)
            
            for w_idx in range(W):
                # For basic T-learner with binary treatment, we usually have w=0,1
                # But implementation should handle W worlds if possible.
                # Here we assume data t matches world indices 0..W-1
                # Or for continuous t, we might need a different approach.
                # The prompt implies standard discrete treatment comparisons.
                
                # Check if we have enough samples for this treatment
                # We use a threshold (e.g. 0.5) to bin if t is standard binary 0/1
                if W == 2:
                    mask_w = (t < 0.5) if w_idx == 0 else (t >= 0.5)
                else:
                    # Exact match for multiple worlds
                    mask_w = np.isclose(t, w_idx, atol=0.1)
                
                X_w = X_cov[mask_w]
                Y_w = Y[mask_w]
                
                if len(Y_w) < 2:
                    # Fallback
                    models[w_idx] = global_mean
                else:
                    model = Ridge(alpha=self.alpha)
                    model.fit(X_w, Y_w)
                    models[w_idx] = model
            
            # Predict
            for w in range(W):
                X_q = query_x_np[b, w, :, :-1]
                model = models.get(w, global_mean)
                
                if isinstance(model, float) or isinstance(model, np.float32):
                    predictions[b, w, :] = model
                else:
                    predictions[b, w, :] = model.predict(X_q)

        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}


class LinearSBaseline:
    """S-Learner using Linear Regression (Ridge)."""

    def __init__(self, device: str = "cpu", alpha: float = 1.0):
        self.device = torch.device(device)
        self.alpha = alpha

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
        """Predicts using a single linear model including treatment as feature."""
        if Ridge is None:
            raise RuntimeError("sklearn is required for LinearBaseline")

        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device

        support_x_np = support_x.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_x_np = query_x.detach().cpu().numpy()

        predictions = np.zeros((B, W, Nq), dtype=np.float32)

        for b in range(B):
            X_sup = support_x_np[b]  # [Ns, d] (includes treatment)
            Y_sup = support_y_np[b]
            
            if support_mask is not None:
                m = support_mask[b].detach().cpu().numpy().flatten().astype(bool)
                if m.sum() < 2:
                    predictions[b, :, :] = np.nanmean(Y_sup) if len(Y_sup) > 0 else 0.0
                    continue
                X_sup = X_sup[m]
                Y_sup = Y_sup[m]
            
            model = Ridge(alpha=self.alpha)
            model.fit(X_sup, Y_sup)
            
            for w in range(W):
                X_q = query_x_np[b, w, :, :] # [Nq, d] (includes treatment set to world w)
                predictions[b, w, :] = model.predict(X_q)

        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
