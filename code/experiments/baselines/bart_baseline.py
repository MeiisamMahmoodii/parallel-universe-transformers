"""BART-style baseline for comparison protocol.

Uses a tree ensemble (GradientBoostingRegressor) to fit E[Y|X,T=0] and E[Y|X,T=1]
on the support set (last feature = treatment). Predicts at each world w using the
model for T=w. Implements the same interface as the model.
Optional: if causalml is installed, could use CausalML's BART; here we use sklearn only.
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None


class BARTBaseline:
    """Predict using tree ensemble (GB) per treatment. Last feature = treatment."""

    def __init__(self, device: str = "cpu", n_estimators: int = 50, max_depth: int = 4):
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
        """Same signature as ParallelUniverseTransformer.forward; returns prediction and log_var."""
        if GradientBoostingRegressor is None:
            raise RuntimeError("sklearn is required for BARTBaseline")
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
                valid = m
                if valid.sum() < 2:
                    predictions[b, :, :] = np.nanmean(Y)
                    continue
                X_cov = X_cov[valid]
                Y = Y[valid]
                t = t[valid]

            X0 = X_cov[t < 0.5]
            Y0 = Y[t < 0.5]
            X1 = X_cov[t >= 0.5]
            Y1 = Y[t >= 0.5]
            mean_y = np.nanmean(Y)

            def fit_gb(X, y):
                if X.shape[0] < 2:
                    return None
                gb = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=42,
                )
                gb.fit(X, y)
                return gb

            gb0 = fit_gb(X0, Y0)
            gb1 = fit_gb(X1, Y1)
            if gb0 is None and gb1 is None:
                predictions[b, :, :] = mean_y
                continue
            if gb0 is None:
                gb0 = gb1
            if gb1 is None:
                gb1 = gb0

            for w in range(W):
                X_q = query_x_np[b, w, :, :-1]
                if w == 0:
                    pred = gb0.predict(X_q)
                else:
                    pred = gb1.predict(X_q)
                predictions[b, w, :] = pred.astype(np.float32)

        if np.isnan(predictions).any():
            predictions = np.nan_to_num(predictions, nan=float(np.nanmean(support_y_np)))
        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
