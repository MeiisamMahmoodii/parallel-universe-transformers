"""PyTorch dataset for streaming SCM episodes."""

from typing import Optional, Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset

from scm.schema import FeatureSchema, SchemaConfig
from scm.sample import SCMSampler, SCMConfig
from scm.intervene import InterventionOperator
from scm.counterfactual import CounterfactualGenerator
from .config import CurriculumStage
from .packer import EpisodePacker, Episode


class SCMEpisodeDataset(IterableDataset):
    """Streaming dataset that generates SCM episodes on-the-fly."""
    
    def __init__(
        self,
        curriculum_stage: CurriculumStage,
        seed: Optional[int] = None,
        episodes_per_epoch: int = 10000,
        rank: int = 0,
        world_size: int = 1
    ):
        """Initialize dataset.
        
        Args:
            curriculum_stage: Configuration for this curriculum stage.
            seed: Random seed.
            episodes_per_epoch: Number of episodes per epoch (for iteration).
            rank: Distributed rank (for DDP sharding).
            world_size: Number of distributed processes (for DDP sharding).
        """
        super().__init__()
        self.stage = curriculum_stage
        self.episodes_per_epoch = episodes_per_epoch
        self.rank = rank
        self.world_size = world_size
        
        # Initialize RNG
        self.base_seed = seed if seed is not None else np.random.randint(0, 2**31)
        self.rng = np.random.RandomState(self.base_seed)
        
        # Initialize components
        self.packer = EpisodePacker(seed=self.rng.randint(0, 2**31))
        self.intervention_operator = InterventionOperator(seed=self.rng.randint(0, 2**31))
    
    def _generate_episode(self, episode_idx: int) -> Episode:
        """Generate a single episode.
        
        Args:
            episode_idx: Index of episode (for seeding).
            
        Returns:
            Episode object.
        """
        # Create episode-specific RNG
        episode_seed = self.base_seed + episode_idx
        episode_rng = np.random.RandomState(episode_seed)
        
        # Sample schema
        schema_config = SchemaConfig(
            n_features=self.stage.n_features,
            n_continuous=self.stage.n_continuous,
            n_categorical=self.stage.n_categorical,
            seed=episode_rng.randint(0, 2**31)
        )
        schema_sampler = FeatureSchema(schema_config)
        schema = schema_sampler.sample_schema()
        
        # Sample SCM
        scm_config = SCMConfig(
            n_features=self.stage.n_features,
            complexity=self.stage.complexity,
            seed=episode_rng.randint(0, 2**31),
            outcome_noise_scale=self.stage.noise_scale
        )
        scm_sampler = SCMSampler(schema, scm_config)
        
        # Sample support set size
        Ns = episode_rng.randint(*self.stage.support_size_range)
        support_x, support_y = scm_sampler.sample(Ns)
        
        # Sample query set size
        Nq = episode_rng.randint(*self.stage.query_size_range)
        query_x_baseline, query_y_baseline = scm_sampler.sample(Nq)
        
        # Sample interventions
        feature_ranges = {
            i: (schema[i].min_value, schema[i].max_value)
            for i in range(len(schema))
            if schema[i].feature_type.value == "continuous"
        }
        
        interventions_list = self.intervention_operator.sample_interventions(
            n_interventions=self.stage.n_interventions,
            n_features=self.stage.n_features,
            feature_ranges=feature_ranges,
            complexity=self.stage.complexity,
            allow_duplicates=False
        )
        
        # Generate counterfactuals
        cf_generator = CounterfactualGenerator(scm_sampler)
        
        # Build worlds: [baseline, intervention_1, ..., intervention_K]
        W = 1 + len(interventions_list)
        query_x_worlds = np.zeros((W, Nq, self.stage.n_features))
        query_y_worlds = np.zeros((W, Nq))
        
        # World 0: baseline
        query_x_worlds[0] = query_x_baseline
        query_y_worlds[0] = query_y_baseline
        
        # Worlds 1..K: interventions
        for i, intervention in enumerate(interventions_list):
            X_cf, Y_cf = cf_generator.generate_counterfactual(query_x_baseline, intervention)
            query_x_worlds[i + 1] = X_cf
            query_y_worlds[i + 1] = Y_cf
        
        # Pack into episode
        interventions_with_baseline = [None] + interventions_list
        episode = self.packer.pack_episode(
            support_x=support_x,
            support_y=support_y,
            query_x_worlds=query_x_worlds,
            query_y_worlds=query_y_worlds,
            schema=schema,
            interventions=interventions_with_baseline,
            missingness_prob=self.stage.missingness_prob
        )
        
        return episode
    
    def __iter__(self) -> Iterator[Episode]:
        """Iterate over episodes. Shards by rank (DDP) then by worker (DataLoader)."""
        # DDP sharding: each rank gets a contiguous slice of episodes
        if self.world_size > 1:
            per_rank = self.episodes_per_epoch // self.world_size
            rank_start = self.rank * per_rank
            rank_end = (self.rank + 1) * per_rank if self.rank < self.world_size - 1 else self.episodes_per_epoch
        else:
            rank_start = 0
            rank_end = self.episodes_per_epoch
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = rank_start
            end_idx = rank_end
        else:
            total_for_rank = rank_end - rank_start
            per_worker = total_for_rank // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = rank_start + worker_id * per_worker
            end_idx = rank_start + (worker_id + 1) * per_worker if worker_id < worker_info.num_workers - 1 else rank_end
        
        for episode_idx in range(start_idx, end_idx):
            yield self._generate_episode(episode_idx)
    
    def __len__(self) -> int:
        """Return number of episodes per epoch."""
        return self.episodes_per_epoch


def create_dataloader(
    curriculum_stage: CurriculumStage,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for SCM episodes.
    
    Args:
        curriculum_stage: Curriculum configuration.
        batch_size: Batch size (per GPU when using DDP).
        num_workers: Number of worker processes.
        seed: Random seed.
        rank: Distributed rank (for DDP; each rank gets a shard of episodes).
        world_size: Number of distributed processes.
        pin_memory: If True, use pinned memory (set False to avoid PyTorch pin_memory device deprecation warnings).
        
    Returns:
        DataLoader instance.
    """
    dataset = SCMEpisodeDataset(
        curriculum_stage,
        seed=seed,
        rank=rank,
        world_size=world_size
    )
    packer = EpisodePacker()
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=packer.collate_episodes,
        pin_memory=pin_memory,
    )
