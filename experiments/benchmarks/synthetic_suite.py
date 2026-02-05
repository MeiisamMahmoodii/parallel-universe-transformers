"""Synthetic benchmark suite for evaluation."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scm.schema import FeatureSchema, SchemaConfig, FeatureType
from scm.sample import SCMSampler, SCMConfig
from scm.intervene import InterventionOperator, InterventionType
from scm.counterfactual import CounterfactualGenerator
from inference.api import ParallelUniverseModel, Intervention
from train.metrics import MetricsComputer
import pandas as pd


class SyntheticBenchmark:
    """Benchmark suite with diverse SCM families."""
    
    def __init__(self, model_path: str):
        """Initialize benchmark.
        
        Args:
            model_path: Path to trained model checkpoint.
        """
        self.model = ParallelUniverseModel.from_pretrained(model_path)
        self.metrics_computer = MetricsComputer()
    
    def generate_test_scm(
        self,
        scm_type: str,
        n_samples: int = 1000,
        seed: int = 42
    ) -> tuple:
        """Generate test data from a specific SCM family.
        
        Args:
            scm_type: Type of SCM ('linear', 'nonlinear', 'multiplicative', etc.).
            n_samples: Number of samples.
            seed: Random seed.
            
        Returns:
            Tuple of (X, Y, scm_sampler, schema).
        """
        # Configure based on type
        if scm_type == 'linear_gaussian':
            complexity = 'simple'
            n_features = 10
        elif scm_type == 'nonlinear_additive':
            complexity = 'moderate'
            n_features = 15
        elif scm_type == 'multiplicative':
            complexity = 'moderate'
            n_features = 15
        elif scm_type == 'heteroskedastic':
            complexity = 'moderate'
            n_features = 15
        elif scm_type == 'heavy_tailed':
            complexity = 'complex'
            n_features = 20
        elif scm_type == 'high_dimensional':
            complexity = 'moderate'
            n_features = 50
        else:
            raise ValueError(f"Unknown SCM type: {scm_type}")
        
        # Create schema
        schema_config = SchemaConfig(
            n_features=n_features,
            n_continuous=n_features // 2,
            n_categorical=n_features // 2,
            seed=seed
        )
        schema_sampler = FeatureSchema(schema_config)
        schema = schema_sampler.sample_schema()
        
        # Create SCM
        scm_config = SCMConfig(
            n_features=n_features,
            complexity=complexity,
            seed=seed + 1
        )
        scm_sampler = SCMSampler(schema, scm_config)
        
        # Generate data
        X, Y = scm_sampler.sample(n_samples)
        
        return X, Y, scm_sampler, schema
    
    def evaluate_scm(
        self,
        scm_type: str,
        n_samples: int = 1000,
        n_interventions: int = 5,
        seed: int = 42
    ) -> Dict:
        """Evaluate model on a specific SCM.
        
        Args:
            scm_type: Type of SCM.
            n_samples: Number of samples.
            n_interventions: Number of interventions to test.
            seed: Random seed.
            
        Returns:
            Dictionary of results.
        """
        print(f"\nEvaluating on {scm_type} SCM...")
        
        # Generate test data
        X, Y, scm_sampler, schema = self.generate_test_scm(scm_type, n_samples, seed)
        
        # Split into support and query
        n_support = n_samples // 2
        support_x, support_y = X[:n_support], Y[:n_support]
        query_x, query_y = X[n_support:], Y[n_support:]
        
        # Sample interventions
        intv_op = InterventionOperator(seed=seed + 2)
        feature_ranges = {
            i: (schema[i].min_value, schema[i].max_value)
            for i in range(len(schema))
            if schema[i].feature_type == FeatureType.CONTINUOUS
        }
        
        interventions = intv_op.sample_interventions(
            n_interventions=n_interventions,
            n_features=len(schema),
            feature_ranges=feature_ranges,
            complexity='moderate'
        )
        
        # Generate ground truth counterfactuals
        cf_generator = CounterfactualGenerator(scm_sampler)
        query_x_cf_batch, query_y_cf_batch = cf_generator.generate_counterfactuals_batch(
            query_x, interventions
        )
        
        # Prepare data for model
        feature_names = [f.name for f in schema]
        support_df = pd.DataFrame(support_x, columns=feature_names)
        query_df = pd.DataFrame(query_x, columns=feature_names)
        
        # Convert interventions to API format
        api_interventions = []
        for intv in interventions:
            if intv.intervention_type == InterventionType.SET:
                intv_type = 'set'
            elif intv.intervention_type == InterventionType.SHIFT:
                intv_type = 'shift'
            else:
                intv_type = 'randomize'
            
            api_interventions.append(Intervention(
                feature=feature_names[intv.feature_idx],
                type=intv_type,
                value=intv.value
            ))
        
        # Predict
        results = self.model.predict_interventions(
            support_df, query_df, api_interventions
        )
        
        # Compute metrics
        baseline_pred = results.baseline
        baseline_true = query_y
        
        cf_pred = results.counterfactuals  # [n_interventions, n_samples]
        cf_true = query_y_cf_batch  # [n_interventions, n_samples]
        
        deltas_pred = results.deltas
        deltas_true = cf_true - baseline_true[None, :]
        
        # Compute errors
        baseline_rmse = np.sqrt(np.mean((baseline_pred - baseline_true) ** 2))
        baseline_mae = np.mean(np.abs(baseline_pred - baseline_true))
        ss_tot_baseline = np.sum((baseline_true - np.mean(baseline_true)) ** 2)
        ss_res_baseline = np.sum((baseline_pred - baseline_true) ** 2)
        baseline_r2 = float(1.0 - ss_res_baseline / (ss_tot_baseline + 1e-8)) if ss_tot_baseline > 1e-8 else 0.0

        cf_rmse = np.sqrt(np.mean((cf_pred - cf_true) ** 2))
        cf_mae = np.mean(np.abs(cf_pred - cf_true))

        delta_rmse = np.sqrt(np.mean((deltas_pred - deltas_true) ** 2))
        delta_mae = np.mean(np.abs(deltas_pred - deltas_true))
        # Delta correlation (flatten and compute Pearson)
        dp_flat = deltas_pred.ravel()
        dt_flat = deltas_true.ravel()
        if np.std(dp_flat) > 1e-8 and np.std(dt_flat) > 1e-8:
            delta_correlation = float(np.corrcoef(dp_flat, dt_flat)[0, 1])
        else:
            delta_correlation = 0.0

        ate_pred = deltas_pred.mean(axis=1)
        ate_true = deltas_true.mean(axis=1)
        ate_mae = np.mean(np.abs(ate_pred - ate_true))

        metrics = {
            'scm_type': scm_type,
            'baseline_rmse': float(baseline_rmse),
            'baseline_mae': float(baseline_mae),
            'baseline_r2': float(baseline_r2),
            'cf_rmse': float(cf_rmse),
            'cf_mae': float(cf_mae),
            'delta_rmse': float(delta_rmse),
            'delta_mae': float(delta_mae),
            'delta_correlation': float(delta_correlation),
            'ate_mae': float(ate_mae),
            'n_samples': n_samples,
            'n_interventions': n_interventions
        }

        print(f"Results: Baseline RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}, R²={baseline_r2:.4f}; "
              f"CF RMSE={cf_rmse:.4f}; Delta RMSE={delta_rmse:.4f}, MAE={delta_mae:.4f}, Corr={delta_correlation:.4f}; "
              f"ATE MAE={ate_mae:.4f}")
        
        return metrics
    
    def run_full_benchmark(self) -> Dict:
        """Run full benchmark suite.
        
        Returns:
            Dictionary of results for all SCM types.
        """
        scm_types = [
            'linear_gaussian',
            'nonlinear_additive',
            'multiplicative',
            'heteroskedastic',
            'heavy_tailed',
            'high_dimensional'
        ]
        
        results = {}
        
        print("\n" + "="*60)
        print("SYNTHETIC BENCHMARK SUITE")
        print("="*60)
        
        for scm_type in scm_types:
            try:
                metrics = self.evaluate_scm(scm_type)
                results[scm_type] = metrics
            except Exception as e:
                print(f"Error evaluating {scm_type}: {e}")
                results[scm_type] = {'error': str(e)}
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for scm_type, metrics in results.items():
            if 'error' not in metrics:
                print(f"\n{scm_type}:")
                print(f"  Baseline RMSE: {metrics['baseline_rmse']:.4f}, MAE: {metrics['baseline_mae']:.4f}, R²: {metrics['baseline_r2']:.4f}")
                print(f"  CF RMSE: {metrics['cf_rmse']:.4f}, MAE: {metrics['cf_mae']:.4f}")
                print(f"  Delta RMSE: {metrics['delta_rmse']:.4f}, MAE: {metrics['delta_mae']:.4f}, Corr: {metrics['delta_correlation']:.4f}")
                print(f"  ATE MAE: {metrics['ate_mae']:.4f}")
        
        return results


def main():
    """Run benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run synthetic benchmark suite")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SyntheticBenchmark(args.checkpoint)
    results = benchmark.run_full_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
