"""Load IHDP (or IHDP-style) data for causal effect evaluation.

IHDP is a standard benchmark: 25 covariates, binary treatment, continuous outcome.
Many versions include potential outcomes (y0, y1) on the test set for PEHE.

Expected CSV columns:
  - Covariates: x1..x25 (or 25 columns before treatment/outcome).
  - Treatment: binary (0/1). Column name can be 't' or 'treatment'.
  - Outcome: observed outcome. Column name 'y' or 'outcome'.
  - Optional (for PEHE): 'y0', 'y1' (potential outcomes). If present, used as test targets.

Train/test split: by default 80/20. Support = train (X, y); query = test.
Query is presented as two worlds: do(T=0) and do(T=1) (covariates with T set to 0 and 1).
"""

import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import urllib.request


# Default URL for one IHDP replication (CEVAE-style CSV if available)
IHDP_SAMPLE_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
)


def load_ihdp_csv(
    path: str,
    treatment_col: str = "t",
    outcome_col: str = "y",
    y0_col: Optional[str] = None,
    y1_col: Optional[str] = None,
    covariate_prefix: str = "x",
    has_y0_y1: bool = True,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load IHDP-style data from CSV.

    Args:
        path: Path to CSV (or URL).
        treatment_col: Name of treatment column.
        outcome_col: Name of outcome column (e.g. 'y' or 'y_factual' for Kaggle).
        y0_col: Name of potential outcome under control (default 'y0'; use 'mu0' for Kaggle).
        y1_col: Name of potential outcome under treatment (default 'y1'; use 'mu1' for Kaggle).
        covariate_prefix: Prefix of covariate columns (e.g. 'x' for x1, x2, ...) or None to use all numeric except t,y,y0,y1.
        has_y0_y1: If True, expect potential-outcome columns (y0/y1 or y0_col/y1_col).
        train_frac: Fraction of rows for support (train); rest for query (test).
        seed: Random seed for train/test split.

    Returns:
        support_x: [Ns, d] float
        support_y: [Ns] float
        query_x_w0: [Nq, d] (query covariates with T=0)
        query_x_w1: [Nq, d] (query covariates with T=1)
        query_y0: [Nq] (true outcome under T=0) if has_y0_y1 else zeros
        query_y1: [Nq] (true outcome under T=1) if has_y0_y1 else zeros
        feature_names: list of d names (order matches columns in support_x)
    """
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
    else:
        df = pd.read_csv(path)

    # Identify columns
    all_cols = list(df.columns)
    if covariate_prefix:
        cov_cols = [c for c in all_cols if c.startswith(covariate_prefix) and c != treatment_col]
        cov_cols = sorted(cov_cols, key=lambda x: (len(x), x))
    if not covariate_prefix or len(cov_cols) == 0:
        skip = {treatment_col, outcome_col, "y0", "y1", "mu0", "mu1", "y_factual", "y_cfactual"}
        cov_cols = [c for c in all_cols if c not in skip and np.issubdtype(df[c].dtype, np.number)]
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not in CSV: {all_cols}")
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not in CSV: {all_cols}")

    feature_names = cov_cols + [treatment_col]
    X = df[cov_cols].values.astype(np.float32)
    t = df[treatment_col].values.astype(np.float32)
    y = df[outcome_col].values.astype(np.float32)
    # Treatment as feature (append column)
    X_with_t = np.hstack([X, t.reshape(-1, 1)])

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_train = max(1, int(n * train_frac))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    support_x = X_with_t[train_idx]
    support_y = y[train_idx]

    X_test = X[test_idx]
    t_test = t[test_idx]
    n_test = len(test_idx)
    query_x_w0 = np.hstack([X_test, np.zeros((n_test, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([X_test, np.ones((n_test, 1), dtype=np.float32)])

    col_y0 = (y0_col if y0_col is not None else "y0") if has_y0_y1 else None
    col_y1 = (y1_col if y1_col is not None else "y1") if has_y0_y1 else None
    if col_y0 and col_y1 and col_y0 in df.columns and col_y1 in df.columns:
        query_y0 = df[col_y0].values[test_idx].astype(np.float32)
        query_y1 = df[col_y1].values[test_idx].astype(np.float32)
    else:
        query_y0 = np.zeros(n_test, dtype=np.float32)
        query_y1 = np.zeros(n_test, dtype=np.float32)

    return support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names


def load_ihdp(
    path: Optional[str] = None,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], bool]:
    """Load IHDP data; use default URL if path is None.

    Returns same as load_ihdp_csv plus has_potential_outcomes: bool.
    """
    if path is None:
        path = IHDP_SAMPLE_URL
    # Try CEVAE-style first: t, y, y0, y1, x1..x25
    try:
        support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names = load_ihdp_csv(
            path,
            treatment_col="t",
            outcome_col="y",
            covariate_prefix="x",
            has_y0_y1=True,
            train_frac=train_frac,
            seed=seed,
        )
        has_potential = np.any(query_y0 != 0) or np.any(query_y1 != 0)
        return support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential
    except Exception:
        pass
    # Try Kaggle-style: treatment, y_factual, mu0, mu1, x1..x25
    try:
        support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names = load_ihdp_csv(
            path,
            treatment_col="treatment",
            outcome_col="y_factual",
            y0_col="mu0",
            y1_col="mu1",
            covariate_prefix="x",
            has_y0_y1=True,
            train_frac=train_frac,
            seed=seed,
        )
        has_potential = np.any(query_y0 != 0) or np.any(query_y1 != 0)
        return support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential
    except Exception:
        pass
    # Fallback: generic column detection
    if path.startswith("http"):
        with urllib.request.urlopen(path) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
    else:
        df = pd.read_csv(path)
    if "treatment" in df.columns and "y_factual" in df.columns:
        treatment_col, outcome_col = "treatment", "y_factual"
        y0_c, y1_c = ("mu0", "mu1") if "mu0" in df.columns and "mu1" in df.columns else (None, None)
    else:
        treatment_col = "t" if "t" in df.columns else "treatment"
        outcome_col = "y" if "y" in df.columns else "y_factual" if "y_factual" in df.columns else "outcome"
        y0_c, y1_c = None, None
    support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names = load_ihdp_csv(
        path,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        y0_col=y0_c,
        y1_col=y1_c,
        covariate_prefix="",
        has_y0_y1=(y0_c is not None and y1_c is not None),
        train_frac=train_frac,
        seed=seed,
    )
    has_potential = np.any(query_y0 != 0) or np.any(query_y1 != 0)
    return support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential
