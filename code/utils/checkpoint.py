"""Checkpoint loading utilities."""

import torch

from model.model import ParallelUniverseTransformer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    ablate_no_cross_world: bool = False,
):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint (.pt).
        device: Device to load model to.
        ablate_no_cross_world: If True, set cross_world_layers=[] at eval (no cross-world attention).

    Returns:
        Model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    cross_world = [] if ablate_no_cross_world else config.get("cross_world_layers", [3, 5])
    model = ParallelUniverseTransformer(
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1024),
        dropout=config.get("dropout", 0.1),
        cross_world_layers=cross_world,
        attend_to_all_worlds=config.get("attend_to_all_worlds", True),
        use_quantiles=config.get("use_quantiles", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
