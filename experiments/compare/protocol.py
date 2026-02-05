"""Comparison protocol specification.

Input:
  - SCM / eval data: curriculum stage (or fixed stage config), seed, num_batches.
  - Equivalent to: (d, n_support, n_query, K) from stage; fixed seed for reproducibility.

Output (per method):
  - Metrics dict: baseline_rmse, baseline_mae, baseline_r2, cf_rmse, cf_mae,
    delta_rmse, delta_mae, delta_correlation, ate_mae, (optional) calibration_ratio, coverage_95.

Methods implement the same interface: given a batch (support_x, support_y, query_x, ...),
return dict with 'prediction' [B, W, Nq] and 'log_var' [B, W, Nq]. Metrics are then
computed via MetricsComputer on (prediction, query_y, log_var, loss_mask).
"""

PROTOCOL_METRICS = [
    "baseline_rmse",
    "baseline_mae",
    "baseline_r2",
    "cf_rmse",
    "cf_mae",
    "cf_r2",
    "delta_rmse",
    "delta_mae",
    "delta_correlation",
    "ate_mae",
    "calibration_ratio",
    "coverage_95",
    "sharpness",
]
