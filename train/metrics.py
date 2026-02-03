"""Evaluation metrics."""

import torch
import numpy as np
from typing import Dict, Optional


class MetricsComputer:
    """Computes evaluation metrics via per-batch accumulation (supports variable-length batches)."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset running accumulators."""
        # Baseline (world 0): RMSE, MAE, R²
        self._sum_sq_err_baseline = 0.0
        self._sum_abs_err_baseline = 0.0
        self._sum_true_baseline = 0.0
        self._sum_true_sq_baseline = 0.0
        self._count_baseline = 0

        # Counterfactual (worlds 1..W)
        self._sum_sq_err_cf = 0.0
        self._sum_abs_err_cf = 0.0
        self._sum_true_cf = 0.0
        self._sum_true_sq_cf = 0.0
        self._count_cf = 0

        # Deltas: RMSE, MAE, correlation (sufficient stats)
        self._sum_sq_err_delta = 0.0
        self._sum_abs_err_delta = 0.0
        self._sum_delta_pred = 0.0
        self._sum_delta_true = 0.0
        self._sum_delta_pred_sq = 0.0
        self._sum_delta_true_sq = 0.0
        self._sum_delta_pred_true = 0.0
        self._count_delta = 0

        # ATE (overall mean delta)
        self._sum_delta_pred_ate = 0.0
        self._sum_delta_true_ate = 0.0
        self._count_ate = 0

        # Calibration: errors, std, coverage, sharpness
        self._sum_sq_err_cal = 0.0
        self._sum_std_cal = 0.0
        self._count_cover_95 = 0
        self._count_cal = 0

    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        log_var: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """Update metrics with a batch using optional mask (1=valid, 0=padded).

        Args:
            y_pred: Predictions of shape [B, W, Nq].
            y_true: Targets of shape [B, W, Nq].
            log_var: Log variance of shape [B, W, Nq].
            loss_mask: Optional mask [B, W, Nq] (1 = valid, 0 = padded).
        """
        if loss_mask is None:
            loss_mask = torch.ones_like(y_pred, device=y_pred.device)

        y_pred = y_pred.detach().cpu().float()
        y_true = y_true.detach().cpu().float()
        log_var = log_var.detach().cpu().float()
        loss_mask = loss_mask.detach().cpu().float()

        # Baseline (world 0)
        m0 = loss_mask[:, 0, :]
        pred0 = y_pred[:, 0, :]
        true0 = y_true[:, 0, :]
        err0 = pred0 - true0
        self._sum_sq_err_baseline += (m0 * err0 ** 2).sum().item()
        self._sum_abs_err_baseline += (m0 * torch.abs(err0)).sum().item()
        self._sum_true_baseline += (m0 * true0).sum().item()
        self._sum_true_sq_baseline += (m0 * true0 ** 2).sum().item()
        self._count_baseline += m0.sum().item()

        # Counterfactual (worlds 1..W)
        if y_pred.shape[1] > 1:
            m_cf = loss_mask[:, 1:, :]
            pred_cf = y_pred[:, 1:, :]
            true_cf = y_true[:, 1:, :]
            err_cf = pred_cf - true_cf
            self._sum_sq_err_cf += (m_cf * err_cf ** 2).sum().item()
            self._sum_abs_err_cf += (m_cf * torch.abs(err_cf)).sum().item()
            self._sum_true_cf += (m_cf * true_cf).sum().item()
            self._sum_true_sq_cf += (m_cf * true_cf ** 2).sum().item()
            self._count_cf += m_cf.sum().item()

        # Deltas
        baseline_pred = y_pred[:, 0:1, :]
        baseline_true = y_true[:, 0:1, :]
        delta_pred = y_pred[:, 1:, :] - baseline_pred
        delta_true = y_true[:, 1:, :] - baseline_true
        m_delta = loss_mask[:, 1:, :]
        err_d = delta_pred - delta_true
        self._sum_sq_err_delta += (m_delta * err_d ** 2).sum().item()
        self._sum_abs_err_delta += (m_delta * torch.abs(err_d)).sum().item()
        self._sum_delta_pred += (m_delta * delta_pred).sum().item()
        self._sum_delta_true += (m_delta * delta_true).sum().item()
        self._sum_delta_pred_sq += (m_delta * delta_pred ** 2).sum().item()
        self._sum_delta_true_sq += (m_delta * delta_true ** 2).sum().item()
        self._sum_delta_pred_true += (m_delta * delta_pred * delta_true).sum().item()
        self._count_delta += m_delta.sum().item()

        # ATE (same as delta means)
        self._sum_delta_pred_ate += (m_delta * delta_pred).sum().item()
        self._sum_delta_true_ate += (m_delta * delta_true).sum().item()
        self._count_ate += m_delta.sum().item()

        # Calibration (all worlds)
        std_pred = torch.exp(0.5 * log_var)
        errors = y_pred - y_true
        z = torch.abs(errors) / (std_pred + 1e-8)
        self._sum_sq_err_cal += (loss_mask * errors ** 2).sum().item()
        self._sum_std_cal += (loss_mask * std_pred).sum().item()
        self._count_cover_95 += (loss_mask * (z < 1.96).float()).sum().item()
        self._count_cal += loss_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute metrics from running accumulators."""
        metrics = {}
        eps = 1e-8

        # Baseline
        if self._count_baseline > 0:
            n = self._count_baseline
            metrics["baseline_rmse"] = np.sqrt(self._sum_sq_err_baseline / n + eps)
            metrics["baseline_mae"] = self._sum_abs_err_baseline / n
            mean_true = self._sum_true_baseline / n
            ss_tot = self._sum_true_sq_baseline - self._sum_true_baseline ** 2 / n
            ss_res = self._sum_sq_err_baseline
            metrics["baseline_r2"] = 1.0 - ss_res / (ss_tot + eps) if ss_tot > eps else 0.0
        else:
            metrics["baseline_rmse"] = float("nan")
            metrics["baseline_mae"] = float("nan")
            metrics["baseline_r2"] = float("nan")

        # Counterfactual
        if self._count_cf > 0:
            n = self._count_cf
            metrics["cf_rmse"] = np.sqrt(self._sum_sq_err_cf / n + eps)
            metrics["cf_mae"] = self._sum_abs_err_cf / n
            mean_true_cf = self._sum_true_cf / n
            ss_tot_cf = self._sum_true_sq_cf - self._sum_true_cf ** 2 / n
            metrics["cf_r2"] = 1.0 - self._sum_sq_err_cf / (ss_tot_cf + eps) if ss_tot_cf > eps else 0.0
        else:
            metrics["cf_rmse"] = float("nan")
            metrics["cf_mae"] = float("nan")
            metrics["cf_r2"] = float("nan")

        # Deltas
        if self._count_delta > 0:
            n = self._count_delta
            metrics["delta_rmse"] = np.sqrt(self._sum_sq_err_delta / n + eps)
            metrics["delta_mae"] = self._sum_abs_err_delta / n
            mean_dp = self._sum_delta_pred / n
            mean_dt = self._sum_delta_true / n
            cov = (self._sum_delta_pred_true / n) - (mean_dp * mean_dt)
            std_dp = np.sqrt(self._sum_delta_pred_sq / n - mean_dp ** 2 + eps)
            std_dt = np.sqrt(self._sum_delta_true_sq / n - mean_dt ** 2 + eps)
            if std_dp > eps and std_dt > eps:
                metrics["delta_correlation"] = cov / (std_dp * std_dt)
            else:
                metrics["delta_correlation"] = 0.0
        else:
            metrics["delta_rmse"] = float("nan")
            metrics["delta_mae"] = float("nan")
            metrics["delta_correlation"] = float("nan")

        # ATE
        if self._count_ate > 0:
            n = self._count_ate
            mean_pred_ate = self._sum_delta_pred_ate / n
            mean_true_ate = self._sum_delta_true_ate / n
            metrics["ate_mae"] = np.abs(mean_pred_ate - mean_true_ate)
        else:
            metrics["ate_mae"] = float("nan")

        # Calibration
        if self._count_cal > 0:
            n = self._count_cal
            metrics["calibration_ratio"] = (self._sum_std_cal / n) / (
                np.sqrt(self._sum_sq_err_cal / n + eps) + eps
            )
            metrics["coverage_95"] = self._count_cover_95 / n
            metrics["sharpness"] = 2.0 * 1.96 * (self._sum_std_cal / n)
        else:
            metrics["calibration_ratio"] = float("nan")
            metrics["coverage_95"] = float("nan")
            metrics["sharpness"] = float("nan")

        return metrics

    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset accumulators."""
        metrics = self.compute()
        self.reset()
        return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for logging."""
    lines = []
    nan = float("nan")

    if "baseline_rmse" in metrics and not np.isnan(metrics.get("baseline_rmse", nan)):
        lines.append(
            f"Baseline - RMSE: {metrics['baseline_rmse']:.4f}, "
            f"MAE: {metrics['baseline_mae']:.4f}, "
            f"R²: {metrics['baseline_r2']:.4f}"
        )
    if "cf_rmse" in metrics and not np.isnan(metrics.get("cf_rmse", nan)):
        lines.append(
            f"Counterfactual - RMSE: {metrics['cf_rmse']:.4f}, "
            f"MAE: {metrics['cf_mae']:.4f}, "
            f"R²: {metrics['cf_r2']:.4f}"
        )
    if "delta_rmse" in metrics and not np.isnan(metrics.get("delta_rmse", nan)):
        lines.append(
            f"Delta - RMSE: {metrics['delta_rmse']:.4f}, "
            f"MAE: {metrics['delta_mae']:.4f}, "
            f"Corr: {metrics['delta_correlation']:.4f}"
        )
    if "ate_mae" in metrics and not np.isnan(metrics.get("ate_mae", nan)):
        lines.append(f"ATE MAE: {metrics['ate_mae']:.4f}")
    if "calibration_ratio" in metrics and not np.isnan(metrics.get("calibration_ratio", nan)):
        lines.append(
            f"Calibration - Ratio: {metrics['calibration_ratio']:.4f}, "
            f"Coverage (95%): {metrics['coverage_95']:.4f}, "
            f"Sharpness: {metrics['sharpness']:.4f}"
        )
    return "\n".join(lines) if lines else "(no metrics)"
