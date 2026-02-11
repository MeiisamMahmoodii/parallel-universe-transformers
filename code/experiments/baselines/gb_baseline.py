"""Gradient Boosting baselines (T-Learner and S-Learner).

T-Learner: Fits two separate GB models, one for treatment group and one for control.
S-Learner: Fits a single GB model with treatment as a feature.
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None


class GBTBaseline:
    """T-Learner using Gradient Boosting."""

    def __init__(self, device: str = "cpu", n_estimators: int = 100, max_depth: int = 3):
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

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
        if GradientBoostingRegressor is None:
            raise RuntimeError("sklearn is required for GBTBaseline")
            
        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device

        support_x_np = support_x.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_x_np = query_x.detach().cpu().numpy()

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

            models = {}
            global_mean = np.nanmean(Y)
            
            for w_idx in range(W):
                if W == 2:
                    mask_w = (t < 0.5) if w_idx == 0 else (t >= 0.5)
                else:
                    mask_w = np.isclose(t, w_idx, atol=0.1)
                
                X_w = X_cov[mask_w]
                Y_w = Y[mask_w]
                
                if len(Y_w) < 2:
                    models[w_idx] = global_mean
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=42
                    )
                    model.fit(X_w, Y_w)
                    models[w_idx] = model
            
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


class GBSBaseline:
    """S-Learner using Gradient Boosting."""

    def __init__(self, device: str = "cpu", n_estimators: int = 100, max_depth: int = 3):
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

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
        if GradientBoostingRegressor is None:
            raise RuntimeError("sklearn is required for GBSBaseline")

        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device

        support_x_np = support_x.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_x_np = query_x.detach().cpu().numpy()

        predictions = np.zeros((B, W, Nq), dtype=np.float32)

        for b in range(B):
            X_sup = support_x_np[b]
            Y_sup = support_y_np[b]
            
            if support_mask is not None:
                m = support_mask[b].detach().cpu().numpy().flatten().astype(bool)
                if m.sum() < 2:
                    predictions[b, :, :] = np.nanmean(Y_sup) if len(Y_sup) > 0 else 0.0
                    continue
                X_sup = X_sup[m]
                Y_sup = Y_sup[m]
            
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            model.fit(X_sup, Y_sup)
            
            for w in range(W):
                X_q = query_x_np[b, w, :, :]
                predictions[b, w, :] = model.predict(X_q)

        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
