"""Metric utilities for evaluation."""

from typing import List, Optional, Tuple

KEY_EVAL_METRICS: List[str] = [
    "delta_correlation",
    "delta_mae",
    "delta_rmse",
    "pehe",
    "ate_mae",
    "delta_slope",
    "delta_intercept",
    "sign_accuracy",
    "baseline_r2",
    "baseline_mae",
    "cf_mae",
    "calibration_ratio",
    "coverage_95",
    "sharpness",
]


def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and standard deviation, ignoring None values.

    Args:
        values: List of values (may contain None).

    Returns:
        (mean, std) or (None, None) if no valid values.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5
