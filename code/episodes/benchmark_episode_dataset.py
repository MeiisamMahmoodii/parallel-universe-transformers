"""Episode builders for ACIC and Twins benchmarks (same format as IHDP)."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from episodes.packer import Episode
from episodes.ihdp_episode_dataset import scale_covariates
from experiments.benchmarks.split_utils import get_finetune_indices
from experiments.benchmarks.acic_data import load_acic_arrays
from experiments.benchmarks.twins_data import load_twins_arrays, DEFAULT_TWINS_DIR


def build_acic_episode(
    path: str,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    max_support: Optional[int] = None,
    max_query: Optional[int] = None,
) -> Optional[Episode]:
    """Build a single ACIC episode: train as support, val as query (W=2: T=0, T=1).

    Args:
        path: Path to ACIC CSV or directory with x.csv + zymu_*.csv.
        train_frac: Training (support) fraction.
        val_frac: Validation (query) fraction for early stopping.
        seed: Random seed for split.

    Returns:
        Episode or None if loading fails.
    """
    try:
        X, t, y, y0, y1, _ = load_acic_arrays(path)
    except Exception:
        return None
    n = len(y)
    support_idx, val_idx, test_idx = get_finetune_indices(
        n, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=0.2
    )
    query_idx = val_idx if len(val_idx) > 0 else test_idx
    rng = np.random.RandomState(seed)
    if max_support is not None and len(support_idx) > max_support:
        take = rng.choice(len(support_idx), max_support, replace=False)
        support_idx = support_idx[take]
    if max_query is not None and len(query_idx) > max_query:
        take = rng.choice(len(query_idx), max_query, replace=False)
        query_idx = query_idx[take]
    x_support_raw = X[support_idx]
    x_query_raw = X[query_idx]
    x_support, x_query = scale_covariates(x_support_raw, x_query_raw)
    t_support = t[support_idx].reshape(-1, 1).astype(np.float32)
    support_x = np.hstack([x_support, t_support]).astype(np.float32)
    support_y = y[support_idx].astype(np.float32)
    nq = len(x_query)
    query_x_w0 = np.hstack([x_query, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([x_query, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([query_x_w0, query_x_w1], axis=0)
    mu0_query = y0[query_idx]
    mu1_query = y1[query_idx]
    query_y = np.stack([mu0_query, mu1_query], axis=0)
    d = support_x.shape[1]
    support_mask = np.zeros((support_x.shape[0], d), dtype=np.float32)
    query_mask = np.zeros((2, nq, d), dtype=np.float32)
    feature_types = np.zeros(d, dtype=np.int64)
    cardinalities = np.ones(d, dtype=np.int64)
    return Episode(
        support_x=torch.from_numpy(support_x).float(),
        support_y=torch.from_numpy(support_y).float(),
        support_mask=torch.from_numpy(support_mask).float(),
        query_x=torch.from_numpy(query_x).float(),
        query_y=torch.from_numpy(query_y).float(),
        query_mask=torch.from_numpy(query_mask).float(),
        feature_types=torch.from_numpy(feature_types).long(),
        cardinalities=torch.from_numpy(cardinalities).long(),
        interventions=[None, None],
    )


def build_twins_episode(
    data_dir: str = DEFAULT_TWINS_DIR,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
    max_support: Optional[int] = None,
    max_query: Optional[int] = None,
) -> Optional[Episode]:
    """Build a single Twins episode: train as support, val as query (W=2: T=0, T=1).

    Args:
        data_dir: Directory containing Final_data_twins.csv.
        train_frac: Training (support) fraction.
        val_frac: Validation (query) fraction for early stopping.
        seed: Random seed for split.
        max_samples: If set, subsample before split (for faster finetuning).

    Returns:
        Episode or None if loading fails.
    """
    try:
        X, t, y, y0, y1, _ = load_twins_arrays(
            data_dir=data_dir, max_samples=max_samples, seed=seed
        )
    except Exception:
        return None
    n = len(y)
    support_idx, val_idx, test_idx = get_finetune_indices(
        n, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=0.2
    )
    query_idx = val_idx if len(val_idx) > 0 else test_idx
    rng = np.random.RandomState(seed)
    if max_support is not None and len(support_idx) > max_support:
        take = rng.choice(len(support_idx), max_support, replace=False)
        support_idx = support_idx[take]
    if max_query is not None and len(query_idx) > max_query:
        take = rng.choice(len(query_idx), max_query, replace=False)
        query_idx = query_idx[take]
    x_support_raw = X[support_idx]
    x_query_raw = X[query_idx]
    x_support, x_query = scale_covariates(x_support_raw, x_query_raw)
    t_support = t[support_idx].reshape(-1, 1).astype(np.float32)
    support_x = np.hstack([x_support, t_support]).astype(np.float32)
    support_y = y[support_idx].astype(np.float32)
    nq = len(x_query)
    query_x_w0 = np.hstack([x_query, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([x_query, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([query_x_w0, query_x_w1], axis=0)
    mu0_query = y0[query_idx]
    mu1_query = y1[query_idx]
    query_y = np.stack([mu0_query, mu1_query], axis=0)
    d = support_x.shape[1]
    support_mask = np.zeros((support_x.shape[0], d), dtype=np.float32)
    query_mask = np.zeros((2, nq, d), dtype=np.float32)
    feature_types = np.zeros(d, dtype=np.int64)
    cardinalities = np.ones(d, dtype=np.int64)
    return Episode(
        support_x=torch.from_numpy(support_x).float(),
        support_y=torch.from_numpy(support_y).float(),
        support_mask=torch.from_numpy(support_mask).float(),
        query_x=torch.from_numpy(query_x).float(),
        query_y=torch.from_numpy(query_y).float(),
        query_mask=torch.from_numpy(query_mask).float(),
        feature_types=torch.from_numpy(feature_types).long(),
        cardinalities=torch.from_numpy(cardinalities).long(),
        interventions=[None, None],
    )
