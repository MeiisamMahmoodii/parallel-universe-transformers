"""Loss functions for training."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_nll_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.
    
    Args:
        y_pred: Predicted values of shape [...].
        y_true: True values of shape [...].
        log_var: Log variance of shape [...].
        reduction: Reduction method ('mean', 'sum', or 'none').
        mask: Optional mask of shape [...] (1 = valid, 0 = ignore). Used for padded positions.
        
    Returns:
        Loss value.
    """
    # NLL = 0.5 * (log(2π) + log_var + (y_true - y_pred)^2 / exp(log_var))
    # We omit the constant log(2π) term
    loss = 0.5 * (log_var + (y_true - y_pred) ** 2 / torch.exp(log_var))
    
    if mask is not None:
        loss = loss * mask
        if reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1)
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def delta_consistency_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Delta consistency loss for effect estimation.
    
    Encourages the model to accurately predict deltas (treatment effects).
    
    Args:
        y_pred: Predicted values of shape [B, W, Nq].
        y_true: True values of shape [B, W, Nq].
        reduction: Reduction method.
        mask: Optional mask [B, W, Nq] (1 = valid, 0 = ignore).
        
    Returns:
        Loss value.
    """
    # Compute deltas (intervention - baseline)
    baseline_pred = y_pred[:, 0:1, :]  # [B, 1, Nq]
    baseline_true = y_true[:, 0:1, :]
    
    delta_pred = y_pred[:, 1:, :] - baseline_pred  # [B, W-1, Nq]
    delta_true = y_true[:, 1:, :] - baseline_true
    
    loss = (delta_pred - delta_true) ** 2
    if mask is not None:
        # mask [B, W, Nq] -> for delta we use mask for worlds 1..W
        delta_mask = mask[:, 1:, :]
        loss = loss * delta_mask
        return loss.sum() / delta_mask.sum().clamp(min=1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def combined_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_var: torch.Tensor,
    lambda_delta: float = 1.0,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None
) -> dict:
    """Combined loss with prediction and delta components.
    
    Args:
        y_pred: Predicted values of shape [B, W, Nq].
        y_true: True values of shape [B, W, Nq].
        log_var: Log variance of shape [B, W, Nq].
        lambda_delta: Weight for delta loss.
        reduction: Reduction method.
        mask: Optional mask [B, W, Nq] for padded positions (1 = valid, 0 = ignore).
        
    Returns:
        Dictionary with 'total', 'pred', and 'delta' losses.
    """
    # Prediction loss (Gaussian NLL)
    loss_pred = gaussian_nll_loss(y_pred, y_true, log_var, reduction, mask)
    
    # Delta consistency loss
    loss_delta = delta_consistency_loss(y_pred, y_true, reduction, mask)
    
    # Total loss
    loss_total = loss_pred + lambda_delta * loss_delta
    
    return {
        'total': loss_total,
        'pred': loss_pred,
        'delta': loss_delta
    }


def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Quantile regression loss (pinball loss).
    
    Args:
        y_pred: Predicted quantiles of shape [B, W, Nq, n_quantiles].
        y_true: True values of shape [B, W, Nq].
        quantiles: Quantile levels of shape [n_quantiles] (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
        reduction: Reduction method.
        
    Returns:
        Loss value.
    """
    # Expand y_true to match quantile dimension
    y_true_expanded = y_true.unsqueeze(-1)  # [B, W, Nq, 1]
    
    # Compute errors
    errors = y_true_expanded - y_pred  # [B, W, Nq, n_quantiles]
    
    # Quantile loss (pinball loss)
    quantiles_expanded = quantiles.view(1, 1, 1, -1)
    loss = torch.max(
        quantiles_expanded * errors,
        (quantiles_expanded - 1) * errors
    )
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class LossComputer:
    """Computes losses for training."""
    
    def __init__(
        self,
        lambda_delta: float = 1.0,
        use_quantiles: bool = False,
        quantile_levels: list = None
    ):
        """Initialize loss computer.
        
        Args:
            lambda_delta: Weight for delta loss.
            use_quantiles: Whether to use quantile loss.
            quantile_levels: Quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
        """
        self.lambda_delta = lambda_delta
        self.use_quantiles = use_quantiles
        
        if quantile_levels is None:
            quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantile_levels = torch.tensor(quantile_levels)
    
    def compute_loss(
        self,
        outputs: dict,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """Compute all losses.
        
        Args:
            outputs: Model outputs dictionary with 'prediction', 'log_var', etc.
            targets: Ground truth values of shape [B, W, Nq].
            loss_mask: Optional mask [B, W, Nq] (1 = valid, 0 = padded). Ignore padded positions.
            
        Returns:
            Dictionary of losses.
        """
        y_pred = outputs['prediction']
        log_var = outputs['log_var']
        
        # Combined loss
        losses = combined_loss(
            y_pred, targets, log_var,
            lambda_delta=self.lambda_delta,
            mask=loss_mask
        )
        
        # Quantile loss (if applicable)
        if self.use_quantiles and 'quantiles' in outputs:
            quantiles = outputs['quantiles']
            self.quantile_levels = self.quantile_levels.to(quantiles.device)
            loss_quantile = quantile_loss(quantiles, targets, self.quantile_levels)
            losses['quantile'] = loss_quantile
            losses['total'] = losses['total'] + 0.1 * loss_quantile  # Small weight
        
        return losses
