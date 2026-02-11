"""Load ACIC 2016 (or ACIC-style) data for causal effect evaluation.

ACIC 2016: semi-synthetic binary treatment, continuous outcome, 58 covariates.
Format: either (1) a single CSV with columns x1..xk, z, y, mu0, mu1 (merged),
or (2) a directory containing x.csv and zymu_*.csv (z=treatment, y=observed, mu0/mu1=potential outcomes).

Same train/support and query/worlds interface as IHDP: support = train (X, y), query = test with two worlds.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List


def load_acic_directory(
    data_dir: str,
    train_frac: float = 0.8,
    seed: int = 42,
    zymu_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], bool]:
    """Load ACIC from directory: x.csv + zymu_*.csv.

    x.csv: covariate matrix (numeric).
    zymu_*.csv: columns z (treatment), y (observed outcome), mu0, mu1 (potential outcomes).

    Returns:
        support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential_outcomes
    """
    data_dir = Path(data_dir)
    x_path = data_dir / "x.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"x.csv not found in {data_dir}")
    x_df = pd.read_csv(x_path)
    # Find zymu file
    if zymu_file:
        zymu_path = data_dir / zymu_file
    else:
        zymu_files = sorted(data_dir.glob("zymu_*.csv"))
        if not zymu_files:
            raise FileNotFoundError(f"No zymu_*.csv in {data_dir}")
        zymu_path = zymu_files[0]
    zymu_df = pd.read_csv(zymu_path)
    if len(x_df) != len(zymu_df):
        raise ValueError(f"Row count mismatch: x.csv {len(x_df)} vs {zymu_path.name} {len(zymu_df)}")
    # Merge by index (same row order)
    df = x_df.copy()
    for c in ["z", "y", "mu0", "mu1"]:
        if c in zymu_df.columns:
            df[c] = zymu_df[c].values
    if "z" not in df.columns or "y" not in df.columns:
        raise ValueError("zymu file must have 'z' and 'y' columns")
    return load_acic_csv(df, train_frac=train_frac, seed=seed)


def load_acic_csv(
    path_or_df,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], bool]:
    """Load ACIC-style data from CSV or DataFrame.

    Expected columns: covariates (x1..xk or all numeric except z,y,mu0,mu1), z (treatment), y (outcome), mu0, mu1 (optional).
    """
    from experiments.benchmarks.ihdp_data import load_ihdp_csv
    if isinstance(path_or_df, (str, Path)):
        path = str(path_or_df)
        df = pd.read_csv(path)
    else:
        df = path_or_df
    # Write to temp CSV so we can reuse load_ihdp_csv (same split logic)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        tmp = f.name
    try:
        support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names = load_ihdp_csv(
            tmp,
            treatment_col="z",
            outcome_col="y",
            y0_col="mu0",
            y1_col="mu1",
            covariate_prefix="",  # auto-detect numeric columns (x1..xk or 0,1,2..)
            has_y0_y1=("mu0" in df.columns and "mu1" in df.columns),
            train_frac=train_frac,
            seed=seed,
        )
    finally:
        os.unlink(tmp)
    has_potential = np.any(query_y0 != 0) or np.any(query_y1 != 0)
    return support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential


def load_acic(
    path: str,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], bool]:
    """Load ACIC data. path can be a CSV file (merged x + zymu) or a directory with x.csv and zymu_*.csv."""
    path = Path(path)
    if path.is_dir():
        return load_acic_directory(path, train_frac=train_frac, seed=seed)
    return load_acic_csv(path, train_frac=train_frac, seed=seed)
