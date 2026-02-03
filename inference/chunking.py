"""Utilities for chunking interventions."""

from typing import List, Iterator
from scm.intervene import Intervention


def chunk_interventions(
    interventions: List[Intervention],
    chunk_size: int
) -> Iterator[List[Intervention]]:
    """Chunk interventions into batches.
    
    Args:
        interventions: List of interventions.
        chunk_size: Size of each chunk.
        
    Yields:
        Chunks of interventions.
    """
    for i in range(0, len(interventions), chunk_size):
        yield interventions[i:i + chunk_size]


def optimize_chunk_size(
    n_interventions: int,
    available_memory_gb: float,
    model_size_gb: float = 1.0,
    batch_size: int = 1
) -> int:
    """Estimate optimal chunk size based on available memory.
    
    Args:
        n_interventions: Total number of interventions.
        available_memory_gb: Available GPU memory in GB.
        model_size_gb: Estimated model size in GB.
        batch_size: Batch size.
        
    Returns:
        Recommended chunk size.
    """
    # Simple heuristic: reserve half for model, half for activations
    usable_memory = (available_memory_gb - model_size_gb) * 0.5
    
    # Estimate memory per intervention (rough approximation)
    memory_per_intervention = 0.1  # GB
    
    max_chunk = int(usable_memory / (memory_per_intervention * batch_size))
    
    # Clamp to reasonable range
    chunk_size = max(1, min(max_chunk, 16))
    
    return chunk_size
