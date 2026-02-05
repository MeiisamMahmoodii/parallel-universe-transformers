"""Baseline stub for comparison protocol.

Implements the same interface as the model forward: given (support_x, support_y, query_x, ...)
returns dict with 'prediction' [B, W, Nq] and 'log_var' [B, W, Nq].
Stub predicts constant baseline = mean(support_y) for all worlds (no intervention effect);
deltas are zero. Use as placeholder until Do-PFN or other baselines are added.
"""

import torch
from typing import Dict, Optional


class MeanBaselineStub:
    """Predicts mean(support_y) for all query positions and all worlds; deltas = 0."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

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
        B, Ns, _ = support_x.shape
        _, W, Nq, _ = query_x.shape
        device = support_x.device
        # Mean of support outcomes per batch item (ignore padding if needed)
        mean_y = support_y.mean(dim=1, keepdim=True)  # [B, 1]
        prediction = mean_y.unsqueeze(2).expand(B, W, Nq).to(device)
        log_var = torch.zeros(B, W, Nq, device=device, dtype=prediction.dtype)
        return {"prediction": prediction, "log_var": log_var}
