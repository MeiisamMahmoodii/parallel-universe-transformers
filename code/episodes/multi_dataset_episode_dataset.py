"""Multi-dataset episode dataset for joint finetuning (IHDP + ACIC + Twins).

Uses dataset-homogeneous batching: each batch contains episodes from only one dataset,
since EpisodePacker.collate_episodes requires same d per batch (IHDP=26, ACIC=59, Twins=69).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from episodes.packer import Episode
from episodes.ihdp_episode_dataset import build_ihdp_episode, DEFAULT_IHDP_DIR, get_ihdp_val_data
from episodes.benchmark_episode_dataset import build_acic_episode, build_twins_episode
from experiments.benchmarks.twins_data import DEFAULT_TWINS_DIR


class MultiDatasetEpisodeDataset(torch.utils.data.Dataset):
    """Dataset that yields episodes from IHDP, ACIC, Twins with configurable mix ratios.

    Flat index space: 0..N1-1 = IHDP, N1..N1+N2-1 = ACIC, N1+N2.. = Twins.
    """

    def __init__(
        self,
        datasets: List[str],
        mix_ratio: List[float],
        size: int,
        seed: int = 42,
        vary_seed: bool = True,
        # IHDP
        ihdp_data_dir: str = DEFAULT_IHDP_DIR,
        # ACIC
        acic_path: Optional[str] = None,
        # Twins
        twins_data_dir: str = DEFAULT_TWINS_DIR,
        twins_max_samples: Optional[int] = None,
        # Common
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        max_support: Optional[int] = None,
        max_query: Optional[int] = None,
    ):
        self.datasets = [d.lower() for d in datasets]
        self.mix_ratio = np.array(mix_ratio, dtype=np.float64)
        self.mix_ratio /= self.mix_ratio.sum()
        self.size = size
        self.seed = seed
        self.vary_seed = vary_seed
        self.ihdp_data_dir = ihdp_data_dir
        self.acic_path = acic_path
        self.twins_data_dir = twins_data_dir
        self.twins_max_samples = twins_max_samples
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.max_support = max_support
        self.max_query = max_query
        # Build cumulative index boundaries: [0, n_ihdp, n_ihdp+n_acic, ...]
        n_per_ds = (self.mix_ratio * size).astype(int)
        n_per_ds[-1] = size - n_per_ds[:-1].sum()
        self._cumsum = np.cumsum([0] + list(n_per_ds))
        self._dataset_ids = []

        for i, ds in enumerate(self.datasets):
            self._dataset_ids.extend([i] * n_per_ds[i])

    def _dataset_id_for_index(self, index: int) -> int:
        return self._dataset_ids[index]

    def _get_episode(self, index: int) -> Episode:
        ds_id = self._dataset_id_for_index(index)
        ds_name = self.datasets[ds_id]
        s = self.seed + index if self.vary_seed else self.seed
        if ds_name == "ihdp":
            ep = build_ihdp_episode(
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=s,
                data_dir=self.ihdp_data_dir,
            )
        elif ds_name == "acic":
            if not self.acic_path:
                raise ValueError("acic_path required when 'acic' in datasets")
            ep = build_acic_episode(
                path=self.acic_path,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=s,
                max_support=self.max_support,
                max_query=self.max_query,
            )
        elif ds_name == "twins":
            ep = build_twins_episode(
                data_dir=self.twins_data_dir,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=s,
                max_samples=self.twins_max_samples,
                max_support=self.max_support,
                max_query=self.max_query,
            )
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")
        if ep is None:
            raise RuntimeError(f"Failed to build episode for {ds_name} at index {index}")
        return ep

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[int, Episode]:
        """Returns (dataset_id, episode) so the batch sampler can group by dataset."""
        ds_id = self._dataset_id_for_index(index)
        return ds_id, self._get_episode(index)


class DatasetHomogeneousBatchSampler(torch.utils.data.Sampler):
    """Samples batches such that all indices in a batch map to the same dataset.

    Yields lists of indices [i1, i2, ...] where all map to the same dataset_id.
    """

    def __init__(
        self,
        dataset: MultiDatasetEpisodeDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._dataset_ids = dataset._dataset_ids

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            rng = np.random.RandomState(self.dataset.seed)
            rng.shuffle(indices)
        # Group by dataset_id
        by_ds: dict = {}
        for i in indices:
            ds_id = self._dataset_ids[i]
            if ds_id not in by_ds:
                by_ds[ds_id] = []
            by_ds[ds_id].append(i)
        # Shuffle dataset order so we don't always do IHDP first
        ds_order = list(by_ds.keys())
        if self.shuffle:
            rng = np.random.RandomState(self.dataset.seed + 1)
            rng.shuffle(ds_order)
        # Yield batches from each dataset
        for ds_id in ds_order:
            group = by_ds[ds_id]
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                if len(batch) >= 1:
                    yield batch

    def __len__(self) -> int:
        total = 0
        by_ds = {}
        for ds_id in self._dataset_ids:
            by_ds[ds_id] = by_ds.get(ds_id, 0) + 1
        for ds_id, count in by_ds.items():
            total += (count + self.batch_size - 1) // self.batch_size
        return total


def _collate_multi_dataset(batch: List[Tuple[int, Episode]]) -> dict:
    """Collate batch of (dataset_id, episode) - drop dataset_id, collate episodes."""
    from episodes.packer import EpisodePacker
    episodes = [ep for _, ep in batch]
    return EpisodePacker().collate_episodes(episodes)


def create_multi_dataset_dataloader(
    datasets: List[str],
    mix_ratio: List[float],
    batch_size: int = 4,
    dataset_size: int = 100,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    ihdp_data_dir: str = DEFAULT_IHDP_DIR,
    acic_path: Optional[str] = None,
    twins_data_dir: str = DEFAULT_TWINS_DIR,
    twins_max_samples: Optional[int] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    max_support: Optional[int] = None,
    max_query: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader that mixes episodes from IHDP, ACIC, Twins.

    Each batch is homogeneous (same dataset) so collate_episodes works.
    """
    dataset = MultiDatasetEpisodeDataset(
        datasets=datasets,
        mix_ratio=mix_ratio,
        size=dataset_size,
        seed=seed,
        vary_seed=True,
        ihdp_data_dir=ihdp_data_dir,
        acic_path=acic_path,
        twins_data_dir=twins_data_dir,
        twins_max_samples=twins_max_samples,
        train_frac=train_frac,
        val_frac=val_frac,
        max_support=max_support,
        max_query=max_query,
    )
    batch_sampler = DatasetHomogeneousBatchSampler(
        dataset, batch_size=batch_size, shuffle=True
    )
    # DataLoader with batch_sampler needs batch_size=1 (sampler specifies batch)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=_collate_multi_dataset,
        pin_memory=pin_memory,
    )


def get_multi_dataset_val_data(
    datasets: List[str],
    seed: int = 42,
    ihdp_data_dir: str = DEFAULT_IHDP_DIR,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
    """Return IHDP val data for early stopping (primary target).

    Uses IHDP validation since it's the main benchmark; ACIC/Twins could be added later.
    """
    if "ihdp" not in datasets:
        return None
    return get_ihdp_val_data(
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        data_dir=ihdp_data_dir,
    )
