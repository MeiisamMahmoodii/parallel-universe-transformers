"""Synthetic benchmark suite for evaluation."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scm.schema import FeatureSchema, SchemaConfig, FeatureType
from scm.sample import SCMSampler, SCMConfig
from scm.intervene import InterventionOperator, InterventionType
from scm.counterfactual import CounterfactualGenerator
from inference.api import ParallelUniverseModel, Intervention
from train.metrics import MetricsComputer

# Import Baselines
from experiments.baselines.linear_baseline import LinearTBaseline, LinearSBaseline
from experiments.baselines.gb_baseline import GBTBaseline, GBSBaseline
from experiments.baselines.dr_baseline import DRBaseline
from experiments.baselines.tabpfn_baseline import TabPFNTBaseline
from experiments.baselines.transtee_baseline import TransTEEBaseline
from experiments.baselines.dragonnet_baseline import DragonnetBaseline
from experiments.baselines.outcome_baseline import OutcomeBaseline


class SyntheticBenchmark:
    """Benchmark suite with diverse SCM families and multiple baselines."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize benchmark.
        
        Args:
            model_path: Path to trained model checkpoint.
        """
        self.device = device
        self.model = ParallelUniverseModel.from_pretrained(model_path)
        self.metrics_computer = MetricsComputer()
        
        # Initialize Baselines
        self.baselines = {
            'Ours': self.model,
            'Linear-T': LinearTBaseline(device=device),
            'Linear-S': LinearSBaseline(device=device),
            'GB-T': GBTBaseline(device=device),
            'GB-S': GBSBaseline(device=device),
            'DR-Linear': DRBaseline(device=device, learner='linear'),
            'DR-GB': DRBaseline(device=device, learner='gb'),
            'TabPFN': TabPFNTBaseline(device=device),
            'TransTEE': TransTEEBaseline(device=device),
            'Dragonnet': DragonnetBaseline(device=device),
            'Ridge': OutcomeBaseline(device=device)
        }
    
    def generate_test_scm(
        self,
        scm_type: str,
        n_samples: int = 1000,
        seed: int = 42
    ) -> tuple:
        """Generate test data from a specific SCM family."""
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
            n_continuous=n_features, # Simplify for baselines (all continuous)
            n_categorical=0,
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
        n_samples: int = 500, # Reduce for speed
        n_interventions: int = 5,
        seed: int = 42
    ) -> Dict:
        """Evaluate all models on a specific SCM."""
        print(f"\nEvaluating on {scm_type} SCM...")
        
        # Generate test data
        X, Y, scm_sampler, schema = self.generate_test_scm(scm_type, n_samples, seed)
        
        # Split into support and query
        n_support = n_samples // 2
        support_x, support_y = X[:n_support], Y[:n_support]
        query_x, query_y = X[n_support:], Y[n_support:]
        
        # Sample interventions (use first continuous feature as treatment)
        # For fair comparison with T-Learners, we treat the LAST feature as treatment T
        # Or we pick one.
        # Our model is general, but baselines expect T column.
        # We will use the standard protocol: Last column = Treatment.
        
        # Strategy: Intervention on Last Feature
        target_feature_idx = len(schema) - 1
        target_feature_name = schema[target_feature_idx].name
        
        # Generate ground truth counterfactuals for T=0 and T=1 (Worlds)
        # World 0: T=0, World 1: T=1
        interventions = [
            Intervention(target_feature_name, 'set', 0.0),
            Intervention(target_feature_name, 'set', 1.0)
        ]
        
        # Use simple internal intervention logic for ground truth
        # Re-using scm logic is best
        from scm.intervene import Intervention as SCMIntervention
        scm_interventions = [
            SCMIntervention(target_feature_idx, InterventionType.SET, 0.0),
            SCMIntervention(target_feature_idx, InterventionType.SET, 1.0)
        ]
        
        cf_generator = CounterfactualGenerator(scm_sampler)
        # CFs for Query set
        # [W, Nq]
        query_y_cfs = []
        for intv in scm_interventions:
            _, y_cf = cf_generator.generate_counterfactuals(query_x, intv)
            query_y_cfs.append(y_cf)
        
        query_y_cfs = np.stack(query_y_cfs, axis=0) # [W, Nq]
        
        # Ground Truth Delta (ATE/CATE)
        true_cate = query_y_cfs[1] - query_y_cfs[0]
        true_ate = np.mean(true_cate)
        
        # Prepare Data for Models
        # Support X needs to be [1, Ns, d]
        # Support Y needs to be [1, Ns]
        # Query X needs to be [1, W, Nq, d]
        
        # Modify Query X to have T=0 and T=1
        query_x_w0 = query_x.copy()
        query_x_w0[:, target_feature_idx] = 0.0
        
        query_x_w1 = query_x.copy()
        query_x_w1[:, target_feature_idx] = 1.0
        
        query_x_stacked = np.stack([query_x_w0, query_x_w1], axis=0) # [W, Nq, d]
        
        # Tensorize
        def to_tensor(x): return torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        t_support_x = to_tensor(support_x)
        t_support_y = to_tensor(support_y)
        t_query_x = to_tensor(query_x_stacked)
        
        # Results container
        scm_results = {}
        
        for name, model in tqdm(self.baselines.items(), desc="Models"):
            try:
                # Predict
                if name == 'Ours':
                    # Our model typically takes dataframe + intervention dicts
                    # But we can assume it also implements the direct forward call
                    # if we want to bypass the API wrapper. 
                    # Actually better to use the forward pass directly for fair speed/interface compare
                    # Model.forward expects:
                    # x, y, query, feature_types, cardinalities
                    
                   pass # Use standard call below
                 
                with torch.no_grad():
                    # Mock feature types (all continuous)
                    ft = torch.zeros(len(schema), dtype=torch.long, device=self.device)
                    cards = torch.zeros(len(schema), dtype=torch.long, device=self.device)
                    
                    if name == 'Ours':
                         # Our model wrapper for direct forward?
                         # The ParallelUniverseModel is a wrapper around the nn.Module
                         # We need the underlying module or call predict_interventions
                         # Let's use the forward if possible or standard interface
                         # Using model.model (the nn.Module)
                         outputs = self.model.model(
                             t_support_x, t_support_y, t_query_x, 
                             ft.unsqueeze(0), cards.unsqueeze(0)
                         )
                         preds = outputs['prediction'].squeeze(0).cpu().numpy() # [W, Nq]
                    else:
                        outputs = model(
                            t_support_x, t_support_y, t_query_x,
                            ft.unsqueeze(0), cards.unsqueeze(0)
                        )
                        preds = outputs['prediction'].squeeze(0).cpu().numpy()
                
                # Compute Metrics
                # Preds: [W, Nq] -> [2, Nq]
                pred_y0 = preds[0]
                pred_y1 = preds[1]
                pred_cate = pred_y1 - pred_y0
                pred_ate = np.mean(pred_cate)
                
                # Errors
                pehe = np.sqrt(np.mean((pred_cate - true_cate) ** 2))
                ate_error = np.abs(pred_ate - true_ate)
                
                # RMSE on outcomes
                rmse_y0 = np.sqrt(np.mean((pred_y0 - query_y_cfs[0]) ** 2))
                rmse_y1 = np.sqrt(np.mean((pred_y1 - query_y_cfs[1]) ** 2))
                avg_rmse = (rmse_y0 + rmse_y1) / 2
                
                scm_results[name] = {
                    'PEHE': float(pehe),
                    'ATE_Err': float(ate_error),
                    'RMSE_Y': float(avg_rmse)
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                scm_results[name] = {'error': str(e)}
                
        return scm_results
    
    def run_full_benchmark(self) -> Dict:
        """Run full benchmark suite."""
        scm_types = [
            'linear_gaussian',
            'nonlinear_additive',
            'multiplicative',
            'heteroskedastic',
             # 'heavy_tailed', # Skip for speed
             # 'high_dimensional'
        ]
        
        results = {}
        
        print("\n" + "="*60)
        print("COMPREHENSIVE BENCHMARK SUITE")
        print("="*60)
        
        for scm_type in scm_types:
            metrics = self.evaluate_scm(scm_type)
            results[scm_type] = metrics
        
        self.print_summary(results)
        return results

    def print_summary(self, results):
        print("\n" + "="*80)
        print(f"{'SCM / Model':<20} | {'PEHE':<10} | {'ATE Err':<10} | {'RMSE Y':<10}")
        print("-" * 80)
        
        for scm, models in results.items():
            print(f"--- {scm.upper()} ---")
            for model, metrics in models.items():
                if 'error' in metrics:
                    print(f"{model:<20} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10}")
                else:
                    print(f"{model:<20} | {metrics['PEHE']:<10.4f} | {metrics['ATE_Err']:<10.4f} | {metrics['RMSE_Y']:<10.4f}")
            print("-" * 80)


def main():
    """Run benchmark suite."""
    import argparse
    parser = argparse.ArgumentParser(description="Run synthetic benchmark suite")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    benchmark = SyntheticBenchmark(args.checkpoint, device=args.device)
    results = benchmark.run_full_benchmark()
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

