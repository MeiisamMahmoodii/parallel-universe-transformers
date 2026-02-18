"""Canonical split utilities for benchmarks.

Ensures finetuning and evaluation use the same seed and split logic so val and test
metrics are comparable. Uses np.random.RandomState(seed) everywhere for reproducibility.
"""

import numpy as np
from typing import Tuple


def get_benchmark_indices(
    n: int,
    seed: int = 42,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get train+val and test indices for evaluation (80/20 split).

    Args:
        n: Total number of samples.
        seed: Random seed.
        test_frac: Fraction for test set (default 0.2).

    Returns:
        train_val_idx: Indices for support (train+val).
        test_idx: Indices for query (test).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    train_val_idx = idx[:-n_test]
    test_idx = idx[-n_test:]
    return train_val_idx, test_idx


def get_finetune_indices(
    n: int,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get train, val, and test indices for finetuning.

    train_frac + val_frac + test_frac should equal 1.0.
    test_idx matches get_benchmark_indices(n, seed, test_frac) so eval uses the same test set.

    Args:
        n: Total number of samples.
        seed: Random seed.
        train_frac: Fraction for training (support).
        val_frac: Fraction for validation (early stopping).
        test_frac: Fraction for test (unused during finetuning; matches eval).

    Returns:
        train_idx: Indices for support during finetuning.
        val_idx: Indices for validation (query during early stopping).
        test_idx: Indices for test (same as evaluation test set).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    n_train_val = n - n_test
    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    # Ensure train + val fits within train_val
    if n_train + n_val > n_train_val:
        n_val = n_train_val - n_train
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val] if n_val > 0 else np.array([], dtype=int)
    test_idx = idx[-n_test:]
    return train_idx, val_idx, test_idx
