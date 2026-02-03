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
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = True,
    eps: float = 1e-6,
):
    """Delta consistency loss for effect estimation.
    
    Encourages the model to accurately predict deltas (treatment effects).
    When scale_invariant=True, loss for backward is normalized by var(delta_true) + eps
    so its scale is comparable to the prediction loss; raw MSE is returned for logging.
    
    Args:
        y_pred: Predicted values of shape [B, W, Nq].
        y_true: True values of shape [B, W, Nq].
        reduction: Reduction method.
        mask: Optional mask [B, W, Nq] (1 = valid, 0 = ignore).
        scale_invariant: If True, normalize by var(delta_true) + eps for gradient.
        eps: Small constant for numerical stability.
        
    Returns:
        (loss_for_total, loss_for_logging): first used in total loss, second for logging (raw MSE).
    """
    # Compute deltas (intervention - baseline)
    baseline_pred = y_pred[:, 0:1, :]  # [B, 1, Nq]
    baseline_true = y_true[:, 0:1, :]
    
    delta_pred = y_pred[:, 1:, :] - baseline_pred  # [B, W-1, Nq]
    delta_true = y_true[:, 1:, :] - baseline_true
    
    sq_err = (delta_pred - delta_true) ** 2
    if mask is not None:
        delta_mask = mask[:, 1:, :]
        n = delta_mask.sum().clamp(min=1)
        loss_raw = (sq_err * delta_mask).sum() / n
        if scale_invariant:
            var_true = (delta_true ** 2 * delta_mask).sum() / n
            scale = var_true + eps
            loss_scaled = (sq_err * delta_mask).sum() / (scale * n)
            return loss_scaled, loss_raw
        return loss_raw, loss_raw
    loss_raw = sq_err.mean() if reduction == "mean" else sq_err.sum()
    if not scale_invariant:
        return loss_raw, loss_raw
    var_true = (delta_true ** 2).mean()
    scale = var_true + eps
    loss_scaled = (sq_err / scale).mean() if reduction == "mean" else (sq_err / scale).sum()
    return loss_scaled, loss_raw


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
    
    # Delta consistency loss (scale-invariant for total, raw MSE for logging)
    loss_delta_for_total, loss_delta_raw = delta_consistency_loss(
        y_pred, y_true, reduction, mask, scale_invariant=True
    )
    
    # Total loss
    loss_total = loss_pred + lambda_delta * loss_delta_for_total
    
    return {
        'total': loss_total,
        'pred': loss_pred,
        'delta': loss_delta_raw
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
        loss_mask: Optional[torch.Tensor] = None,
        lambda_delta_override: Optional[float] = None,
    ) -> dict:
        """Compute all losses.
        
        Args:
            outputs: Model outputs dictionary with 'prediction', 'log_var', etc.
            targets: Ground truth values of shape [B, W, Nq].
            loss_mask: Optional mask [B, W, Nq] (1 = valid, 0 = padded). Ignore padded positions.
            lambda_delta_override: If set, use this instead of self.lambda_delta (e.g. for warmup).
            
        Returns:
            Dictionary of losses.
        """
        y_pred = outputs['prediction']
        log_var = outputs['log_var']
        lambda_delta = lambda_delta_override if lambda_delta_override is not None else self.lambda_delta
        
        # Combined loss
        losses = combined_loss(
            y_pred, targets, log_var,
            lambda_delta=lambda_delta,
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
