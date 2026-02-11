"""Episode packer for converting SCM samples to model-ready tensors."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import torch

from scm.schema import FeatureInfo, FeatureType
from scm.intervene import Intervention


@dataclass
class Episode:
    """A single training episode with support set, query set, and multiple worlds."""
    
    # Support set (observational data for conditioning)
    support_x: torch.Tensor  # [Ns, d]
    support_y: torch.Tensor  # [Ns]
    support_mask: torch.Tensor  # [Ns, d] - missingness indicators
    
    # Query set (baseline + interventions)
    query_x: torch.Tensor  # [W, Nq, d] - W worlds
    query_y: torch.Tensor  # [W, Nq] - ground truth per world
    query_mask: torch.Tensor  # [W, Nq, d]
    
    # Metadata
    feature_types: torch.Tensor  # [d] - 0=continuous, 1=categorical
    cardinalities: torch.Tensor  # [d] - cardinality for each feature
    interventions: List[Optional[Intervention]]  # [W] - None for world 0 (baseline)
    
    def to(self, device: torch.device):
        """Move episode to device."""
        return Episode(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            support_mask=self.support_mask.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            query_mask=self.query_mask.to(device),
            feature_types=self.feature_types.to(device),
            cardinalities=self.cardinalities.to(device),
            interventions=self.interventions,
        )


class EpisodePacker:
    """Packs SCM samples into Episode objects."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def pack_episode(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x_worlds: np.ndarray,  # [W, Nq, d]
        query_y_worlds: np.ndarray,  # [W, Nq]
        schema: List[FeatureInfo],
        interventions: List[Optional[Intervention]],
        missingness_prob: float = 0.0
    ) -> Episode:
        """Pack arrays into an Episode object.
        
        Args:
            support_x: Support features [Ns, d].
            support_y: Support outcomes [Ns].
            query_x_worlds: Query features for all worlds [W, Nq, d].
            query_y_worlds: Query outcomes for all worlds [W, Nq].
            schema: List of FeatureInfo objects.
            interventions: List of interventions (None for baseline world).
            missingness_prob: Probability of missing values.
            
        Returns:
            Episode object.
        """
        Ns, d = support_x.shape
        W, Nq, _ = query_x_worlds.shape
        
        # Create missingness masks
        support_mask = self._create_missingness_mask(Ns, d, missingness_prob)
        query_mask = self._create_missingness_mask(W * Nq, d, missingness_prob).reshape(W, Nq, d)
        
        # Apply missingness (set to 0 where mask is 1)
        support_x_masked = support_x.copy()
        support_x_masked[support_mask == 1] = 0
        
        query_x_masked = query_x_worlds.copy()
        for w in range(W):
            query_x_masked[w][query_mask[w] == 1] = 0
        
        # Extract feature metadata
        feature_types = np.array([
            0 if f.feature_type == FeatureType.CONTINUOUS else 1
            for f in schema
        ])
        cardinalities = np.array([
            1 if f.feature_type == FeatureType.CONTINUOUS else f.cardinality
            for f in schema
        ])
        
        # Convert to tensors
        return Episode(
            support_x=torch.from_numpy(support_x_masked).float(),
            support_y=torch.from_numpy(support_y).float(),
            support_mask=torch.from_numpy(support_mask).float(),
            query_x=torch.from_numpy(query_x_masked).float(),
            query_y=torch.from_numpy(query_y_worlds).float(),
            query_mask=torch.from_numpy(query_mask).float(),
            feature_types=torch.from_numpy(feature_types).long(),
            cardinalities=torch.from_numpy(cardinalities).long(),
            interventions=interventions,
        )
    
    def _create_missingness_mask(
        self,
        n_samples: int,
        n_features: int,
        missingness_prob: float
    ) -> np.ndarray:
        """Create missingness mask.
        
        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            missingness_prob: Probability of missing value.
            
        Returns:
            Binary mask of shape [n_samples, n_features] where 1=missing.
        """
        if missingness_prob == 0:
            return np.zeros((n_samples, n_features), dtype=np.float32)
        
        return (self.rng.rand(n_samples, n_features) < missingness_prob).astype(np.float32)
    
    def collate_episodes(self, episodes: List[Episode]) -> Dict[str, torch.Tensor]:
        """Collate a batch of episodes with padding for variable Ns/Nq.
        
        Episodes can have different support set sizes (Ns) and query set sizes (Nq).
        We pad to max Ns and max Nq in the batch; padded positions get mask=1 so
        the model can ignore them (use attention mask downstream).
        
        Args:
            episodes: List of Episode objects.
            
        Returns:
            Dictionary of batched tensors (padded) plus support_lengths, query_lengths.
        """
        B = len(episodes)
        d = episodes[0].support_x.shape[1]
        W = episodes[0].query_x.shape[0]
        
        max_Ns = max(ep.support_x.shape[0] for ep in episodes)
        max_Nq = max(ep.query_x.shape[1] for ep in episodes)
        
        support_lengths = [ep.support_x.shape[0] for ep in episodes]
        query_lengths = [ep.query_x.shape[1] for ep in episodes]
        
        # Allocate padded tensors
        support_x = torch.zeros(B, max_Ns, d, dtype=episodes[0].support_x.dtype)
        support_y = torch.zeros(B, max_Ns, dtype=episodes[0].support_y.dtype)
        support_mask = torch.ones(B, max_Ns, d, dtype=episodes[0].support_mask.dtype)
        
        query_x = torch.zeros(B, W, max_Nq, d, dtype=episodes[0].query_x.dtype)
        query_y = torch.zeros(B, W, max_Nq, dtype=episodes[0].query_y.dtype)
        query_mask = torch.ones(B, W, max_Nq, d, dtype=episodes[0].query_mask.dtype)
        
        for i, ep in enumerate(episodes):
            Ns, Nq = ep.support_x.shape[0], ep.query_x.shape[1]
            support_x[i, :Ns, :] = ep.support_x
            support_y[i, :Ns] = ep.support_y
            support_mask[i, :Ns, :] = ep.support_mask
            support_mask[i, Ns:, :] = 1.0  # padded = missing
            
            query_x[i, :, :Nq, :] = ep.query_x
            query_y[i, :, :Nq] = ep.query_y
            query_mask[i, :, :Nq, :] = ep.query_mask
            query_mask[i, :, Nq:, :] = 1.0  # padded = missing
        
        batch = {
            'support_x': support_x,
            'support_y': support_y,
            'support_mask': support_mask,
            'query_x': query_x,
            'query_y': query_y,
            'query_mask': query_mask,
            'feature_types': torch.stack([ep.feature_types for ep in episodes]),
            'cardinalities': torch.stack([ep.cardinalities for ep in episodes]),
            'support_lengths': torch.tensor(support_lengths, dtype=torch.long),
            'query_lengths': torch.tensor(query_lengths, dtype=torch.long),
            'max_Ns': max_Ns,
            'max_Nq': max_Nq,
        }
        batch['interventions'] = [ep.interventions for ep in episodes]
        
        return batch
