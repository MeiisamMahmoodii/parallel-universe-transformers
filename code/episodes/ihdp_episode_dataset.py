"""IHDP dataset that yields Episode(s) for fine-tuning (same format as SCM episodes)."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from episodes.packer import Episode
from experiments.benchmarks.split_utils import get_finetune_indices


def scale_covariates(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scale covariates with StandardScaler (fit on train, transform both). Reduces distribution mismatch vs SCM training."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)
    return x_train_s, x_test_s


def scale_outcomes(
    y_support: np.ndarray,
    mu0_query: np.ndarray,
    mu1_query: np.ndarray,
):
    """Scale outcomes with StandardScaler (fit on support Y, transform support and query).
    Returns (y_support_s, query_y_s, scaler). query_y_s has shape (2, Nq) for worlds 0,1."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    y_support_s = scaler.fit_transform(y_support.reshape(-1, 1)).flatten().astype(np.float32)
    mu0_s = scaler.transform(mu0_query.reshape(-1, 1)).flatten().astype(np.float32)
    mu1_s = scaler.transform(mu1_query.reshape(-1, 1)).flatten().astype(np.float32)
    query_y_s = np.stack([mu0_s, mu1_s], axis=0)
    return y_support_s, query_y_s, scaler

# IHDP default path (relative to repo root when running from repo root)
DEFAULT_IHDP_DIR = "data/ihdp"
DEFAULT_IHDP_FILE = "ihdp_npci_1.csv"
URL_IHDP = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"


def load_ihdp_arrays(data_dir: str = DEFAULT_IHDP_DIR, file_name: str = DEFAULT_IHDP_FILE):
    """Load IHDP CSV; return x, t, y, mu0, mu1 (all numpy)."""
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL_IHDP, path)
    data = pd.read_csv(path, header=None)
    t = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    mu0 = data.iloc[:, 3].values.astype(np.float32)
    mu1 = data.iloc[:, 4].values.astype(np.float32)
    x = data.iloc[:, 5:].values.astype(np.float32)
    return x, t, y, mu0, mu1


def build_ihdp_episode(
    train_frac: float = 0.8,
    val_frac: float = 0.0,
    seed: int = 42,
    data_dir: str = DEFAULT_IHDP_DIR,
    scale_outcome: bool = False,
) -> Optional[Episode]:
    """Build a single IHDP episode: train as support, val and/or test as query (W=2: T=0, T=1).
    If val_frac>0: train_frac is support, val_frac is query (for training+early stop); rest is test (unused).
    Query outcomes are mu0, mu1 so the model can learn CATE.
    If scale_outcome: fit StandardScaler on support_y, scale support_y and query_y for training.
    """
    x, t, y, mu0, mu1 = load_ihdp_arrays(data_dir=data_dir)
    n = len(y)
    support_idx, val_idx, test_idx = get_finetune_indices(
        n, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=0.2
    )
    query_idx = val_idx if len(val_idx) > 0 else test_idx
    # Scale covariates (fit on support, transform support and query) to reduce distribution mismatch
    x_support_raw = x[support_idx]
    x_query_raw = x[query_idx]
    x_support, x_query = scale_covariates(x_support_raw, x_query_raw)
    # Support: (X, T) -> Y
    t_support = t[support_idx].reshape(-1, 1).astype(np.float32)
    y_support = y[support_idx].astype(np.float32)
    support_x = np.hstack([x_support, t_support])
    # Query: val or test set with two worlds (T=0, T=1); outcomes = mu0, mu1
    nq = len(x_query)
    query_x_w0 = np.hstack([x_query, np.zeros((nq, 1), dtype=np.float32)])
    query_x_w1 = np.hstack([x_query, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([query_x_w0, query_x_w1], axis=0)  # [2, Nq, d]
    mu0_query = mu0[query_idx]
    mu1_query = mu1[query_idx]
    if scale_outcome:
        support_y, query_y, _ = scale_outcomes(y_support, mu0_query, mu1_query)
    else:
        support_y = y_support
        query_y = np.stack([mu0_query, mu1_query], axis=0)  # [2, Nq]
    d = support_x.shape[1]
    # No missingness
    support_mask = np.zeros((support_x.shape[0], d), dtype=np.float32)
    query_mask = np.zeros((2, nq, d), dtype=np.float32)
    # All continuous
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


def get_ihdp_val_data(
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    data_dir: str = DEFAULT_IHDP_DIR,
    scale_outcome: bool = False,
) -> Optional[Tuple]:
    """Return (support_x, support_y, query_x, true_cate) or (..., scaler) when scale_outcome.
    support = train_frac, query = val_frac; true_cate = mu1 - mu0 on val (always in original scale).
    """
    x, t, y, mu0, mu1 = load_ihdp_arrays(data_dir=data_dir)
    n = len(y)
    support_idx, val_idx, test_idx = get_finetune_indices(
        n, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=0.2
    )
    if len(val_idx) <= 0:
        return None
    x_support_raw, x_val_raw = x[support_idx], x[val_idx]
    x_support, x_val = scale_covariates(x_support_raw, x_val_raw)
    t_support = t[support_idx].reshape(-1, 1).astype(np.float32)
    support_x = np.hstack([x_support, t_support]).astype(np.float32)
    y_support = y[support_idx].astype(np.float32)
    mu0_val = mu0[val_idx]
    mu1_val = mu1[val_idx]
    true_cate = (mu1_val - mu0_val).astype(np.float32)
    if scale_outcome:
        support_y, _, scaler = scale_outcomes(y_support, mu0_val, mu1_val)
    else:
        support_y = y_support
        scaler = None
    nq = len(x_val)
    q0 = np.hstack([x_val, np.zeros((nq, 1), dtype=np.float32)])
    q1 = np.hstack([x_val, np.ones((nq, 1), dtype=np.float32)])
    query_x = np.stack([q0, q1], axis=0).astype(np.float32)
    result = (
        torch.from_numpy(support_x).float(),
        torch.from_numpy(support_y).float(),
        torch.from_numpy(query_x).float(),
        true_cate,
    )
    if scaler is not None:
        return result + (scaler,)
    return result


class IHDPEpisodeDataset(torch.utils.data.Dataset):
    """Dataset that returns IHDP episodes for fine-tuning (real, synthetic, or mixed)."""

    def __init__(
        self,
        data_dir: str = DEFAULT_IHDP_DIR,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        size: int = 50,
        seed: int = 42,
        vary_seed: bool = False,
        use_synthetic: bool = False,
        synthetic_mode: str = "bootstrap",
        synthetic_mix_ratio: float = 0.0,
        synthetic_noise_std: float = 0.5,
        scale_outcome: bool = False,
    ):
        self.data_dir = data_dir
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.size = size
        self.seed = seed
        self.vary_seed = vary_seed
        self.use_synthetic = use_synthetic
        self.synthetic_mode = synthetic_mode
        self.synthetic_mix_ratio = synthetic_mix_ratio
        self.synthetic_noise_std = synthetic_noise_std
        self.scale_outcome = scale_outcome
        self._episode: Optional[Episode] = None

    def _get_episode(self, index: int) -> Episode:
        from episodes.synthetic_ihdp_generator import (
            generate_ihdp_bootstrap_episode,
            generate_ihdp_linear_episode,
        )

        use_syn = self.use_synthetic
        if self.synthetic_mix_ratio > 0 and not self.use_synthetic:
            rng = np.random.RandomState(self.seed + index)
            use_syn = rng.rand() < self.synthetic_mix_ratio

        if use_syn:
            s = self.seed + index if self.vary_seed else self.seed
            if self.synthetic_mode == "bootstrap":
                return generate_ihdp_bootstrap_episode(
                    n_samples=747,
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    seed=s,
                    data_dir=self.data_dir,
                    noise_std=self.synthetic_noise_std,
                )
            elif self.synthetic_mode == "linear":
                return generate_ihdp_linear_episode(
                    n_samples=747,
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    seed=s,
                )
            else:
                return generate_ihdp_bootstrap_episode(
                    n_samples=747,
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    seed=s,
                    data_dir=self.data_dir,
                    noise_std=self.synthetic_noise_std,
                )

        if self._episode is None or self.vary_seed:
            s = self.seed + index if self.vary_seed else self.seed
            ep = build_ihdp_episode(
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=s,
                data_dir=self.data_dir,
                scale_outcome=self.scale_outcome,
            )
            if not self.vary_seed:
                self._episode = ep
            return ep
        return self._episode

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Episode:
        return self._get_episode(index)


def create_ihdp_dataloader(
    data_dir: str = DEFAULT_IHDP_DIR,
    batch_size: int = 4,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    dataset_size: int = 50,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_synthetic: bool = False,
    synthetic_mode: str = "bootstrap",
    synthetic_mix_ratio: float = 0.0,
    synthetic_noise_std: float = 0.5,
    scale_outcome: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader of IHDP episodes for fine-tuning."""
    from episodes.packer import EpisodePacker
    dataset = IHDPEpisodeDataset(
        data_dir=data_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        size=dataset_size,
        seed=seed,
        vary_seed=use_synthetic or synthetic_mix_ratio > 0,
        use_synthetic=use_synthetic,
        synthetic_mode=synthetic_mode,
        synthetic_mix_ratio=synthetic_mix_ratio,
        synthetic_noise_std=synthetic_noise_std,
        scale_outcome=scale_outcome,
    )
    packer = EpisodePacker()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=packer.collate_episodes,
        pin_memory=pin_memory,
    )
