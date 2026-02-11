"""CausalPFN baseline (amortized causal effect estimation via in-context learning).

Uses causalpfn.CATEEstimator: fit(X, T, Y), estimate_cate(X_query).
"""

import torch
import numpy as np
from typing import Dict, Optional
import warnings

try:
    from causalpfn import CATEEstimator
except ImportError:
    CATEEstimator = None


class CausalPFNBaseline:
    """CATE estimation via CausalPFN (fit on support, predict CATE at query)."""

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = torch.device(device)
        self.verbose = verbose

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
        if CATEEstimator is None:
            warnings.warn("CausalPFN not installed. Returning zero predictions.")
            B, _, Nq, _ = query_x.shape
            W = query_x.shape[1]
            return {
                "prediction": torch.zeros(B, W, Nq, device=self.device),
                "log_var": torch.zeros(B, W, Nq, device=self.device),
            }

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

            # CausalPFN: fit on (X, T, Y), predict CATE at query X
            X_q = query_x_np[b, 0, :, :-1]  # query covariates (same for T=0 and T=1)

            try:
                est = CATEEstimator(device=str(self.device), verbose=self.verbose)
                est.fit(X_cov.astype(np.float32), t.astype(np.float32), Y.astype(np.float32))
                cate_hat = est.estimate_cate(X_q.astype(np.float32))
                if np.isscalar(cate_hat):
                    cate_hat = np.full(Nq, float(cate_hat))
                cate_hat = np.asarray(cate_hat, dtype=np.float32).flatten()
                if cate_hat.size != Nq:
                    cate_hat = np.broadcast_to(np.mean(cate_hat), Nq).copy()
                # Output: pred_y0 = 0, pred_y1 = CATE so that pred_y1 - pred_y0 = CATE
                predictions[b, 0, :] = 0.0
                predictions[b, 1, :] = cate_hat[:Nq]
            except Exception as e:
                warnings.warn(f"CausalPFN failed: {e}")
                mean_y = np.nanmean(Y)
                predictions[b, 0, :] = mean_y
                predictions[b, 1, :] = mean_y

        prediction = torch.from_numpy(predictions).to(device=device, dtype=support_x.dtype)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
