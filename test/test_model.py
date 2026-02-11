"""Tests for model components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import torch
import numpy as np

from model.embeddings import ContinuousEncoder, CategoricalEncoder
from model.tokenizer import TabularTokenizer
from model.backbone import TransformerEncoder
from model.model import ParallelUniverseTransformer


def test_continuous_encoder():
    """Test continuous feature encoder."""
    encoder = ContinuousEncoder(d_model=64, n_fourier=8)
    
    x = torch.randn(10, 5)
    encoded = encoder(x)
    
    assert encoded.shape == (10, 5, 64)


def test_categorical_encoder():
    """Test categorical feature encoder."""
    encoder = CategoricalEncoder(d_model=64, max_cardinality=20)
    
    x = torch.randint(0, 10, (10, 5))
    cardinalities = torch.tensor([10] * 5)
    
    encoded = encoder(x, cardinalities)
    
    assert encoded.shape == (10, 5, 64)


def test_tokenizer():
    """Test tabular tokenizer."""
    tokenizer = TabularTokenizer(d_model=64, max_features=10)
    
    B, Ns, Nq, d, W = 2, 8, 4, 5, 3
    
    support_x = torch.randn(B, Ns, d)
    support_y = torch.randn(B, Ns)
    query_x = torch.randn(B, W, Nq, d)
    feature_types = torch.zeros(d, dtype=torch.long)  # All continuous
    cardinalities = torch.ones(d, dtype=torch.long)
    
    support_tokens, query_tokens = tokenizer(
        support_x, support_y, query_x,
        feature_types, cardinalities
    )
    
    # Check shapes
    assert support_tokens.shape[0] == B
    assert support_tokens.shape[2] == 64  # d_model
    assert query_tokens.shape[0] == B
    assert query_tokens.shape[1] == W
    assert query_tokens.shape[3] == 64


def test_transformer_encoder():
    """Test transformer encoder."""
    encoder = TransformerEncoder(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        cross_world_layers=[1]
    )
    
    B, W, N = 2, 3, 20
    Ns, Nq = 10, 10
    
    x = torch.randn(B * W, N, 64)
    
    output = encoder(x, W, Ns, Nq)
    
    assert output.shape == (B * W, N, 64)


def test_full_model():
    """Test full model forward pass."""
    model = ParallelUniverseTransformer(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_features=10,
        cross_world_layers=[1]
    )
    
    B, Ns, Nq, d, W = 2, 8, 4, 5, 3
    
    support_x = torch.randn(B, Ns, d)
    support_y = torch.randn(B, Ns)
    query_x = torch.randn(B, W, Nq, d)
    feature_types = torch.zeros(d, dtype=torch.long)
    cardinalities = torch.ones(d, dtype=torch.long)
    
    outputs = model(
        support_x, support_y, query_x,
        feature_types, cardinalities
    )
    
    assert 'prediction' in outputs
    assert 'log_var' in outputs
    assert 'deltas' in outputs
    
    assert outputs['prediction'].shape == (B, W, Nq)
    assert outputs['log_var'].shape == (B, W, Nq)
    assert outputs['deltas'].shape == (B, W - 1, Nq)


if __name__ == '__main__':
    print("Running model tests...")
    
    test_continuous_encoder()
    print("✓ Continuous encoder test passed")
    
    test_categorical_encoder()
    print("✓ Categorical encoder test passed")
    
    test_tokenizer()
    print("✓ Tokenizer test passed")
    
    test_transformer_encoder()
    print("✓ Transformer encoder test passed")
    
    test_full_model()
    print("✓ Full model test passed")
    
    print("\nAll tests passed!")
