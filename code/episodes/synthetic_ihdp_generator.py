"""Synthetic IHDP episode generators for training augmentation.

Approach A (bootstrap): Resample from real IHDP, add Gaussian noise to outcomes.
Approach B (linear): Generate (X, T, Y, mu0, mu1) from a linear DGP.
"""

from typing import Optional

import numpy as np
import torch

from episodes.ihdp_episode_dataset import (
    DEFAULT_IHDP_DIR,
    load_ihdp_arrays,
    scale_covariates,
)
from episodes.packer import Episode


def generate_ihdp_bootstrap_episode(
    n_samples: int = 747,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    data_dir: str = DEFAULT_IHDP_DIR,
    noise_std: float = 0.5,
) -> Episode:
    """Build an IHDP episode by bootstrapping from real data with outcome noise.

    Resamples (X, T) from real IHDP with replacement; adds Gaussian noise to mu0/mu1
    to produce y0, y1; observed y = (1-T)*y0 + T*y1. Creates diverse outcome
    realizations while keeping covariate distribution.
    """
    x, t, _, mu0, mu1 = load_ihdp_arrays(data_dir=data_dir)
    n_total = len(t)
    rng = np.random.RandomState(seed)

    # Bootstrap: sample indices with replacement
    idx = rng.randint(0, n_total, size=n_samples)
    x_boot = x[idx].astype(np.float32)
    t_boot = t[idx].astype(np.float32)
    mu0_boot = mu0[idx].astype(np.float32)
    mu1_boot = mu1[idx].astype(np.float32)

    # Add noise to potential outcomes
    eps0 = rng.randn(n_samples).astype(np.float32) * noise_std
    eps1 = rng.randn(n_samples).astype(np.float32) * noise_std
    y0 = mu0_boot + eps0
    y1 = mu1_boot + eps1
    y_boot = (1 - t_boot) * y0 + t_boot * y1

    # Split into support and query
    n_support = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples) if val_frac > 0 else 0
    perm = rng.permutation(n_samples)
    support_idx = perm[:n_support]
    val_idx = perm[n_support : n_support + n_val] if n_val > 0 else np.array([], dtype=int)
    query_idx = val_idx if n_val > 0 else perm[n_support:]

    x_support_raw = x_boot[support_idx]
    x_query_raw = x_boot[query_idx]
    x_support, x_query = scale_covariates(x_support_raw, x_query_raw)

    t_support = t_boot[support_idx].reshape(-1, 1).astype(np.float32)
    y_support = y_boot[support_idx].astype(np.float32)
    support_x = np.hstack([x_support, t_support])
    support_y = y_support

    nq = len(x_query)
    query_x_w0 = np.hstack([x_query, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([x_query, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([query_x_w0, query_x_w1], axis=0)

    y0_query = y0[query_idx]
    y1_query = y1[query_idx]
    query_y = np.stack([y0_query, y1_query], axis=0)

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


def generate_ihdp_linear_episode(
    n_samples: int = 747,
    n_covariates: int = 25,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Episode:
    """Generate an IHDP-style episode from a linear DGP.

    X ~ N(0, 1), T ~ Bernoulli(0.5), mu0 = X @ beta0, mu1 = X @ beta1 + offset,
    Y = (1-T)*y0 + T*y1 + noise. Matches IHDP structure (25 dims, binary T, continuous Y).
    """
    rng = np.random.RandomState(seed)
    n_support = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples) if val_frac > 0 else 0
    n_query = n_samples - n_support - n_val
    if n_query <= 0 and n_val <= 0:
        n_query = n_samples - n_support

    x = rng.randn(n_samples, n_covariates).astype(np.float32)
    t = (rng.rand(n_samples) < 0.5).astype(np.float32)
    beta0 = rng.randn(n_covariates).astype(np.float32) * 0.3
    beta1 = rng.randn(n_covariates).astype(np.float32) * 0.3
    offset = float(rng.randn() * 0.5)

    mu0 = x @ beta0
    mu1 = x @ beta1 + offset
    eps = rng.randn(n_samples).astype(np.float32) * 0.5
    y0 = mu0 + eps
    y1 = mu1 + eps
    y = (1 - t) * y0 + t * y1

    perm = rng.permutation(n_samples)
    support_idx = perm[:n_support]
    val_idx = perm[n_support : n_support + n_val] if n_val > 0 else np.array([], dtype=int)
    query_idx = val_idx if n_val > 0 else perm[n_support:]

    x_support_raw = x[support_idx]
    x_query_raw = x[query_idx]
    x_support, x_query = scale_covariates(x_support_raw, x_query_raw)

    t_support = t[support_idx].reshape(-1, 1).astype(np.float32)
    y_support = y[support_idx].astype(np.float32)
    support_x = np.hstack([x_support, t_support])
    support_y = y_support

    nq = len(x_query)
    query_x_w0 = np.hstack([x_query, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([x_query, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([query_x_w0, query_x_w1], axis=0)

    mu0_query = mu0[query_idx]
    mu1_query = mu1[query_idx]
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
