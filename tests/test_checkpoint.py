"""Tests for checkpoint loading and model forward sanity.

Run with a checkpoint available:
  CHECKPOINT_PATH=checkpoints/checkpoint_step_5000.pt pytest tests/test_checkpoint.py -v

If CHECKPOINT_PATH is unset or file missing, checkpoint tests are skipped.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import ParallelUniverseTransformer
from episodes.config import CurriculumConfig
from episodes.dataset import create_dataloader
from episodes.packer import EpisodePacker


def _get_checkpoint_path():
    path = os.environ.get("CHECKPOINT_PATH", "")
    if path and Path(path).exists():
        return path
    return None


def _load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Build model from checkpoint config and load weights (mirrors inference/api.py)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = ParallelUniverseTransformer(
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1024),
        dropout=config.get("dropout", 0.1),
        cross_world_layers=config.get("cross_world_layers", [3, 5]),
        attend_to_all_worlds=config.get("attend_to_all_worlds", True),
        use_quantiles=config.get("use_quantiles", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def test_checkpoint_load_and_forward():
    """Load checkpoint, build model, run one batch, assert shapes and delta consistency."""
    checkpoint_path = _get_checkpoint_path()
    if checkpoint_path is None:
        import pytest
        pytest.skip("CHECKPOINT_PATH not set or file missing; skipping checkpoint test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    # One curriculum stage, one batch
    stages = CurriculumConfig.get_default_curriculum()
    stage = stages[0]
    dataloader = create_dataloader(
        stage,
        batch_size=1,
        num_workers=0,
        seed=42,
        rank=0,
        world_size=1,
        pin_memory=False,  # avoid PyTorch pin_memory(device) deprecation warnings
    )

    batch = next(iter(dataloader))
    support_x = batch["support_x"].to(device)
    support_y = batch["support_y"].to(device)
    query_x = batch["query_x"].to(device)
    feature_types = batch["feature_types"][0].to(device)
    cardinalities = batch["cardinalities"][0].to(device)
    support_mask = batch["support_mask"].to(device)
    query_mask = batch["query_mask"].to(device)

    with torch.no_grad():
        out = model(
            support_x,
            support_y,
            query_x,
            feature_types,
            cardinalities,
            support_mask,
            query_mask,
        )

    prediction = out["prediction"]
    log_var = out["log_var"]
    deltas = out["deltas"]

    B, W, Nq = prediction.shape
    assert B >= 1 and W >= 2 and Nq >= 1, f"Unexpected shapes B={B} W={W} Nq={Nq}"

    # Shapes
    assert prediction.shape == (B, W, Nq), f"prediction shape {prediction.shape}"
    assert log_var.shape == (B, W, Nq), f"log_var shape {log_var.shape}"
    assert deltas.shape == (B, W - 1, Nq), f"deltas shape {deltas.shape}"

    # No NaN/Inf
    assert torch.isfinite(prediction).all(), "prediction has non-finite values"
    assert torch.isfinite(log_var).all(), "log_var has non-finite values"
    assert torch.isfinite(deltas).all(), "deltas has non-finite values"

    # Delta consistency: deltas = prediction[:,1:,:] - prediction[:,0:1,:]
    baseline = prediction[:, 0:1, :]
    cf = prediction[:, 1:, :]
    expected_deltas = cf - baseline
    tol = 1e-5
    assert torch.allclose(deltas, expected_deltas, atol=tol, rtol=tol), (
        "deltas should equal counterfactuals minus baseline"
    )
