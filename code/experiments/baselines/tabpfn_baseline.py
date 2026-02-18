"""TabPFN T-Learner baseline.

Uses TabPFNRegressor to fit T=0 and T=1 models.
"""

import os
import torch
import numpy as np
from typing import Dict, Optional
import warnings

# Allow TabPFN to run on CPU with >1000 samples (e.g. Twins, ACIC)
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None


class TabPFNTBaseline:
    """T-Learner using TabPFN."""

    def __init__(self, device: str = "cpu", N_ensemble_configurations: int = 3):
        self.device = torch.device(device)
        self.N_ensemble_configurations = N_ensemble_configurations

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
        if TabPFNRegressor is None:
            # Return dummy predictions if TabPFN is not installed
            # This allows the benchmark to run (with error reporting) without crashing
            warnings.warn("TabPFN not installed. Returning zero predictions.")
            B, _, Nq, _ = query_x.shape
            W = query_x.shape[1]
            return {
                "prediction": torch.zeros(B, W, Nq, device=self.device),
                "log_var": torch.zeros(B, W, Nq, device=self.device)
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

            models = {}
            global_mean = np.nanmean(Y)
            
            # T-Learner fit
            for w_idx in range(W):
                if W == 2:
                    mask_w = (t < 0.5) if w_idx == 0 else (t >= 0.5)
                else:
                    mask_w = np.isclose(t, w_idx, atol=0.1)
                
                X_w = X_cov[mask_w]
                Y_w = Y[mask_w]
                
                # TabPFN is efficient for small datasets, but check minimums
                if len(Y_w) < 5: # Arbitrary small number
                    models[w_idx] = global_mean
                else:
                    # TabPFN expects cpu numpy
                    # We might need to subsample if N is too large (TabPFN limit usually 1000 or so)
                    if len(Y_w) > 1024:
                        indices = np.random.choice(len(Y_w), 1024, replace=False)
                        X_w = X_w[indices]
                        Y_w = Y_w[indices]
                        
                    model = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
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
        # TabPFN doesn't output variance by default in basic predict, 
        # though it can does full posterior. For baseline, we leave log_var 0.
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
