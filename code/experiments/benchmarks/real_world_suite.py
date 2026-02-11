"""Real-world benchmark suite (IHDP).

Downloads and evaluates on IHDP (Infant Health and Development Program) dataset.
Standard benchmark for CATE estimation.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
import urllib.request
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
from experiments.baselines.causalpfn_baseline import CausalPFNBaseline
from episodes.ihdp_episode_dataset import scale_covariates

class IHDPDataset:
    """Loader for IHDP dataset."""
    
    URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    
    def __init__(self, data_dir: str = "data/ihdp"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.file_path = os.path.join(data_dir, "ihdp_npci_1.csv")
    
    def download(self):
        if not os.path.exists(self.file_path):
            print(f"Downloading IHDP dataset to {self.file_path}...")
            urllib.request.urlretrieve(self.URL, self.file_path)
            print("Download complete.")
            
    def load(self):
        self.download()
        # The CSV has headers: treatment, y_factual, y_cf, mu0, mu1, x1...x25
        # Note: The raw file usually doesn't have headers or has specific format.
        # Let's inspect or assume standard format.
        # Based on CEVAE report: first col is treatment, second is y, third is y_cf, fourth is mu0, fifth is mu1.
        
        try:
           data = pd.read_csv(self.file_path, header=None)
        except Exception as e:
            print(f"Error loading IHDP: {e}")
            return None, None, None, None
            
        # Columns:
        # 0: treatment
        # 1: y_factual
        # 2: y_cf
        # 3: mu0
        # 4: mu1
        # 5-29: x (25 covars)
        
        t = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        y_cf = data.iloc[:, 2].values
        mu0 = data.iloc[:, 3].values
        mu1 = data.iloc[:, 4].values
        x = data.iloc[:, 5:].values
        
        # Calculate true CATE
        cate = mu1 - mu0
        
        return x, t, y, cate


class RealWorldBenchmark:
    """Benchmark suite for real-world datasets."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.baselines = {
            'Linear-T': LinearTBaseline(device=device),
            'Linear-S': LinearSBaseline(device=device),
            'GB-T': GBTBaseline(device=device),
            'GB-S': GBSBaseline(device=device),
            'DR-Linear': DRBaseline(device=device, learner='linear'),
            'DR-GB': DRBaseline(device=device, learner='gb'),
            'TabPFN': TabPFNTBaseline(device=device),
            'CausalPFN': CausalPFNBaseline(device=device),
            'TransTEE': TransTEEBaseline(device=device),
            'Dragonnet': DragonnetBaseline(device=device),
            'Ridge': OutcomeBaseline(device=device)
        }
        if model_path and Path(model_path).exists():
            self.model = ParallelUniverseModel.from_pretrained(model_path, device=device)
            self.baselines = {'Ours': self.model, **self.baselines}
        else:
            self.model = None
    
    def evaluate_ihdp(self) -> Dict:
        """Evaluate on IHDP."""
        print("\nEvaluating on IHDP Dataset...")
        dataset = IHDPDataset()
        x, t, y, true_cate = dataset.load()
        
        if x is None:
            return {'error': 'Failed to load dataset'}
        
        # IHDP is small (747 samples). We can use 80/20 train/test.
        # Baselines usually train on Train and Predict on Test.
        # For our "One-Shot" / "In-Context" model, we put Train in Support and Test in Query.
        
        n_samples = len(y)
        n_train = int(0.8 * n_samples)
        
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        x_train, t_train, y_train = x[train_idx], t[train_idx], y[train_idx]
        x_test, t_test, _ = x[test_idx], t[test_idx], y[test_idx]
        true_cate_test = true_cate[test_idx]
        # Scale covariates (fit on train) to reduce distribution mismatch with SCM training
        x_train, x_test = scale_covariates(x_train, x_test)
        
        # Construct Support and Query
        # Support: (X, T) -> Y
        # For our model, input is (X with T appended), Y.
        # Query: (X with T=0) and (X with T=1).
        
        # Support X needs T as last column
        support_x = np.hstack([x_train, t_train.reshape(-1, 1)])
        support_y = y_train
        
        # Query X needs T=0 and T=1
        x_test_0 = np.hstack([x_test, np.zeros((len(x_test), 1))])
        x_test_1 = np.hstack([x_test, np.ones((len(x_test), 1))])
        
        query_x = np.stack([x_test_0, x_test_1], axis=0) # [W=2, Nq, d]
        
        # Tensorize
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        t_support_x = to_tensor(support_x)
        t_support_y = to_tensor(support_y)
        t_query_x = to_tensor(query_x)
        
        results = {}
        
        for name, model in tqdm(self.baselines.items(), desc="IHDP Models"):
            try:
                # Dummy feature types
                ft = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device)
                cards = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device)
                
                if name == 'Ours':
                    with torch.no_grad():
                        outputs = self.model.model(
                            t_support_x, t_support_y, t_query_x, 
                            ft.unsqueeze(0), cards.unsqueeze(0)
                        )
                        preds = outputs['prediction'].squeeze(0).cpu().numpy()
                else:
                    # Baselines (TransTEE/Dragonnet) need grads for internal training
                    # So we DO NOT use torch.no_grad() here.
                    outputs = model(
                        t_support_x, t_support_y, t_query_x,
                        ft.unsqueeze(0), cards.unsqueeze(0)
                    )
                    preds = outputs['prediction'].squeeze(0).cpu().numpy()
                
                # Preds: [2, Nq]
                pred_y0 = preds[0]
                pred_y1 = preds[1]
                pred_cate = pred_y1 - pred_y0
                
                pehe = np.sqrt(np.mean((pred_cate - true_cate_test) ** 2))
                ate_error = np.abs(np.mean(pred_cate) - np.mean(true_cate_test))
                
                results[name] = {
                    'PEHE': float(pehe),
                    'ATE_Err': float(ate_error)
                }
            except Exception as e:
                print(f"Error {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results

    def print_summary(self, results):
        print("\n" + "="*60)
        print("IHDP RESULTS")
        print("="*60)
        print(f"{'Model':<20} | {'PEHE':<10} | {'ATE Err':<10}")
        print("-" * 60)
        for model, metrics in results.items():
             if 'error' in metrics:
                 print(f"{model:<20} | {'ERROR':<10} | {'ERROR':<10}")
             else:
                 print(f"{model:<20} | {metrics['PEHE']:<10.4f} | {metrics['ATE_Err']:<10.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to our model checkpoint (optional; if missing, only baselines run)')
    parser.add_argument('--output', type=str, default='ihdp_results.json')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if not args.checkpoint or not Path(args.checkpoint).exists():
        print("Note: No checkpoint provided or file missing. Running baselines only (Ours skipped).")
    benchmark = RealWorldBenchmark(args.checkpoint, device=args.device)
    results = benchmark.evaluate_ihdp()
    benchmark.print_summary(results)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
