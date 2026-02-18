"""Load Twins dataset for causal effect evaluation.

Twins: twin births dataset from NBER, adapted for observational CATE benchmarking.
Binary treatment (heavier twin mortality), outcome (mortality), covariates.
Has y0, y1 for PEHE.

Source: https://github.com/shalit-lab/Benchmarks (Final_data_twins.csv)
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

TWINS_URL = "https://github.com/shalit-lab/Benchmarks/raw/master/Twins/Final_data_twins.csv"
DEFAULT_TWINS_DIR = "data/twins"


def load_twins_arrays(
    data_dir: str = DEFAULT_TWINS_DIR,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load Twins raw arrays (X, t, y, y0, y1, feature_names) without splitting."""
    os.makedirs(data_dir, exist_ok=True)
    path = Path(data_dir) / "Final_data_twins.csv"
    if not path.exists():
        urllib.request.urlretrieve(TWINS_URL, path)
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]], errors="ignore")
    skip_cols = {"T", "y0", "y1", "yf", "y_cf", "Propensity", "y_factual"}
    cov_cols = [c for c in df.columns if c not in skip_cols and np.issubdtype(df[c].dtype, np.number)]
    if not cov_cols:
        cov_cols = [c for c in df.columns if c not in {"T", "y0", "y1", "yf", "y_cf", "Propensity"}]
    t_col = "T" if "T" in df.columns else "treatment"
    y_col = "yf" if "yf" in df.columns else ("y_factual" if "y_factual" in df.columns else "y")
    y0_col = "y0" if "y0" in df.columns else "mu0"
    y1_col = "y1" if "y1" in df.columns else "mu1"
    X = df[cov_cols].values.astype(np.float32)
    t = df[t_col].values.astype(np.float32)
    y = df[y_col].values.astype(np.float32)
    y0 = df[y0_col].values.astype(np.float32) if y0_col in df.columns else np.zeros(len(y), dtype=np.float32)
    y1 = df[y1_col].values.astype(np.float32) if y1_col in df.columns else np.zeros(len(y), dtype=np.float32)
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        take = rng.permutation(len(X))[:max_samples]
        X, t, y, y0, y1 = X[take], t[take], y[take], y0[take], y1[take]
    feature_names = cov_cols + [t_col]
    return X, t, y, y0, y1, feature_names


def load_twins(
    data_dir: str = DEFAULT_TWINS_DIR,
    train_frac: float = 0.8,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load Twins CSV; return support/query arrays for W=2 evaluation.

    Args:
        max_samples: If set, subsample to this many rows before split (for faster benchmarking).

    Returns:
        support_x: [Ns, d] with T appended, scaled
        support_y: [Ns]
        query_x_w0: [Nq, d] (T=0), scaled
        query_x_w1: [Nq, d] (T=1), scaled
        query_y0: [Nq]
        query_y1: [Nq]
        feature_names: list of covariate + treatment names
    """
    from episodes.ihdp_episode_dataset import scale_covariates
    from experiments.benchmarks.split_utils import get_benchmark_indices

    X, t, y, y0, y1, feature_names = load_twins_arrays(
        data_dir=data_dir, max_samples=max_samples, seed=seed
    )
    n = len(X)
    train_val_idx, test_idx = get_benchmark_indices(n, seed=seed, test_frac=1.0 - train_frac)

    X_train, X_test = X[train_val_idx], X[test_idx]
    t_train = t[train_val_idx]
    y_train = y[train_val_idx]
    y0_test = y0[test_idx]
    y1_test = y1[test_idx]

    X_train, X_test = scale_covariates(X_train, X_test)

    support_x = np.hstack([X_train, t_train.reshape(-1, 1)]).astype(np.float32)
    support_y = y_train.astype(np.float32)
    nq = len(X_test)
    query_x_w0 = np.hstack([X_test, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([X_test, np.ones((nq, 1), dtype=np.float32)])

    return support_x, support_y, query_x_w0, query_x_w1, y0_test, y1_test, feature_names
