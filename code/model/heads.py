"""Prediction and uncertainty heads."""

import torch
import torch.nn as nn
from typing import Optional


class PredictionHead(nn.Module):
    """Prediction head for regression outcomes."""
    
    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1):
        """Initialize prediction head.
        
        Args:
            d_model: Input dimension.
            hidden_dim: Hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict outcomes.
        
        Args:
            x: Hidden states of shape [B*W, Nq, d_model].
            
        Returns:
            Predictions of shape [B*W, Nq, 1].
        """
        return self.net(x)


class UncertaintyHead(nn.Module):
    """Uncertainty head for Gaussian NLL (predicts log variance)."""
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0
    ):
        """Initialize uncertainty head.
        
        Args:
            d_model: Input dimension.
            hidden_dim: Hidden dimension.
            dropout: Dropout probability.
            min_log_var: Minimum log variance (for stability).
            max_log_var: Maximum log variance (for stability).
        """
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log variance.
        
        Args:
            x: Hidden states of shape [B*W, Nq, d_model].
            
        Returns:
            Log variance of shape [B*W, Nq, 1].
        """
        log_var = self.net(x)
        # Clamp for stability
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        return log_var


class QuantileHead(nn.Module):
    """Quantile prediction head for uncertainty quantification."""
    
    def __init__(
        self,
        d_model: int,
        n_quantiles: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """Initialize quantile head.
        
        Args:
            d_model: Input dimension.
            n_quantiles: Number of quantiles to predict.
            hidden_dim: Hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_quantiles = n_quantiles
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_quantiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantiles.
        
        Args:
            x: Hidden states of shape [B*W, Nq, d_model].
            
        Returns:
            Quantiles of shape [B*W, Nq, n_quantiles].
        """
        return self.net(x)


class DeltaHead(nn.Module):
    """Explicit delta (effect) head: predicts intervention effect from baseline vs intervention representations."""

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Input: concat of baseline and intervention repr -> 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        baseline_hidden: torch.Tensor,
        intervention_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Predict deltas (effect = intervention - baseline) per query.

        Args:
            baseline_hidden: [B, Nq, d_model] - baseline world query Y-token hidden states.
            intervention_hidden: [B, W-1, Nq, d_model] - intervention worlds query Y-token hidden states.

        Returns:
            Deltas of shape [B, W-1, Nq].
        """
        B, Nq, d = baseline_hidden.shape
        _, Wm1, _, _ = intervention_hidden.shape
        # Expand baseline to match each intervention: [B, W-1, Nq, d_model]
        base = baseline_hidden.unsqueeze(1).expand(B, Wm1, Nq, d)
        concat = torch.cat([base, intervention_hidden], dim=-1)
        out = self.net(concat).squeeze(-1)
        return out


class CombinedHead(nn.Module):
    """Combined prediction and uncertainty head."""
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_quantiles: bool = False,
        n_quantiles: int = 5
    ):
        """Initialize combined head.
        
        Args:
            d_model: Input dimension.
            hidden_dim: Hidden dimension.
            dropout: Dropout probability.
            use_quantiles: Whether to use quantile head.
            n_quantiles: Number of quantiles (if use_quantiles=True).
        """
        super().__init__()
        self.use_quantiles = use_quantiles
        
        self.prediction_head = PredictionHead(d_model, hidden_dim, dropout)
        self.uncertainty_head = UncertaintyHead(d_model, hidden_dim, dropout)
        
        if use_quantiles:
            self.quantile_head = QuantileHead(d_model, n_quantiles, hidden_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Predict outcomes and uncertainty.
        
        Args:
            x: Hidden states of shape [B*W, Nq, d_model].
            
        Returns:
            Dictionary with:
                - 'prediction': [B*W, Nq, 1]
                - 'log_var': [B*W, Nq, 1]
                - 'quantiles': [B*W, Nq, n_quantiles] (if use_quantiles=True)
        """
        output = {
            'prediction': self.prediction_head(x),
            'log_var': self.uncertainty_head(x)
        }
        
        if self.use_quantiles:
            output['quantiles'] = self.quantile_head(x)
        
        return output
