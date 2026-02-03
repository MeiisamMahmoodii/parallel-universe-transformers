"""Tests for SCM components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scm.schema import FeatureSchema, SchemaConfig, FeatureType
from scm.sample import SCMSampler, SCMConfig
from scm.intervene import InterventionOperator, Intervention, InterventionType
from scm.counterfactual import CounterfactualGenerator


def test_schema_generation():
    """Test feature schema generation."""
    config = SchemaConfig(
        n_features=10,
        n_continuous=5,
        n_categorical=5,
        seed=42
    )
    
    schema_sampler = FeatureSchema(config)
    schema = schema_sampler.sample_schema()
    
    assert len(schema) == 10
    
    # Count feature types
    n_continuous = sum(1 for f in schema if f.feature_type == FeatureType.CONTINUOUS)
    n_categorical = sum(1 for f in schema if f.feature_type == FeatureType.CATEGORICAL)
    
    assert n_continuous == 5
    assert n_categorical == 5


def test_scm_sampling():
    """Test SCM data generation."""
    schema_config = SchemaConfig(n_features=5, n_continuous=3, n_categorical=2, seed=42)
    schema_sampler = FeatureSchema(schema_config)
    schema = schema_sampler.sample_schema()
    
    scm_config = SCMConfig(n_features=5, complexity='simple', seed=43)
    scm_sampler = SCMSampler(schema, scm_config)
    
    X, Y = scm_sampler.sample(100)
    
    assert X.shape == (100, 5)
    assert Y.shape == (100,)
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()


def test_interventions():
    """Test intervention operator."""
    intv_op = InterventionOperator(seed=42)
    
    # Create sample data
    X = np.random.randn(10, 5)
    
    # Test SET intervention
    intervention = Intervention(feature_idx=2, intervention_type=InterventionType.SET, value=1.0)
    X_do = intv_op.apply(X, intervention)
    
    assert X_do.shape == X.shape
    assert np.all(X_do[:, 2] == 1.0)
    
    # Test SHIFT intervention
    intervention = Intervention(feature_idx=1, intervention_type=InterventionType.SHIFT, value=0.5)
    X_do = intv_op.apply(X, intervention)
    
    assert np.allclose(X_do[:, 1], X[:, 1] + 0.5)


def test_counterfactuals():
    """Test counterfactual generation."""
    schema_config = SchemaConfig(n_features=5, n_continuous=3, n_categorical=2, seed=42)
    schema_sampler = FeatureSchema(schema_config)
    schema = schema_sampler.sample_schema()
    
    scm_config = SCMConfig(n_features=5, complexity='simple', seed=43)
    scm_sampler = SCMSampler(schema, scm_config)
    
    X, Y = scm_sampler.sample(10)
    
    # Create intervention
    intervention = Intervention(feature_idx=2, intervention_type=InterventionType.SET, value=0.5)
    
    # Generate counterfactuals
    cf_generator = CounterfactualGenerator(scm_sampler)
    X_cf, Y_cf = cf_generator.generate_counterfactual(X, intervention)
    
    assert X_cf.shape == X.shape
    assert Y_cf.shape == Y.shape
    assert np.all(X_cf[:, 2] == 0.5)


if __name__ == '__main__':
    print("Running SCM tests...")
    test_schema_generation()
    print("✓ Schema generation test passed")
    
    test_scm_sampling()
    print("✓ SCM sampling test passed")
    
    test_interventions()
    print("✓ Interventions test passed")
    
    test_counterfactuals()
    print("✓ Counterfactuals test passed")
    
    print("\nAll tests passed!")
