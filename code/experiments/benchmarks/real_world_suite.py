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
from experiments.benchmarks.twins_data import load_twins
from experiments.benchmarks.acic_data import load_acic
from experiments.benchmarks.split_utils import get_benchmark_indices

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
    
    def evaluate_ihdp(self, seed: int = 42, scale_outcome_for_ours: bool = False) -> Dict:
        """Evaluate on IHDP with given random seed for train/test split."""
        print("\nEvaluating on IHDP Dataset...")
        dataset = IHDPDataset()
        x, t, y, true_cate = dataset.load()
        
        if x is None:
            return {'error': 'Failed to load dataset'}
        
        # IHDP is small (747 samples). We use 80/20 train/test (same split as split_utils).
        n_samples = len(y)
        train_val_idx, test_idx = get_benchmark_indices(n_samples, seed=seed, test_frac=0.2)
        
        x_train, t_train, y_train = x[train_val_idx], t[train_val_idx], y[train_val_idx]
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
        scaler = None
        if scale_outcome_for_ours:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(support_y.reshape(-1, 1))

        # Query X needs T=0 and T=1
        x_test_0 = np.hstack([x_test, np.zeros((len(x_test), 1))])
        x_test_1 = np.hstack([x_test, np.ones((len(x_test), 1))])
        
        query_x = np.stack([x_test_0, x_test_1], axis=0) # [W=2, Nq, d]
        
        # Tensorize
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        support_y_scaled = scaler.transform(support_y.reshape(-1, 1)).flatten() if scaler is not None else None
        t_support_x = to_tensor(support_x)
        t_query_x = to_tensor(query_x)
        
        results = {}
        
        for name, model in tqdm(self.baselines.items(), desc="IHDP Models"):
            try:
                # Dummy feature types
                ft = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device)
                cards = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device)
                
                use_scaled = (name == 'Ours' and scaler is not None)
                sy = to_tensor(support_y_scaled) if use_scaled else to_tensor(support_y)
                
                if name == 'Ours':
                    with torch.no_grad():
                        outputs = self.model.model(
                            t_support_x, sy, t_query_x, 
                            ft.unsqueeze(0), cards.unsqueeze(0)
                        )
                        preds = outputs['prediction'].squeeze(0).cpu().numpy()
                else:
                    # Baselines (TransTEE/Dragonnet) need grads for internal training
                    # So we DO NOT use torch.no_grad() here.
                    outputs = model(
                        t_support_x, sy, t_query_x,
                        ft.unsqueeze(0), cards.unsqueeze(0)
                    )
                    preds = outputs['prediction'].squeeze(0).cpu().numpy()
                
                # Preds: [2, Nq]
                pred_y0 = preds[0]
                pred_y1 = preds[1]
                if use_scaled and scaler is not None:
                    pred_y0 = scaler.inverse_transform(pred_y0.reshape(-1, 1)).flatten()
                    pred_y1 = scaler.inverse_transform(pred_y1.reshape(-1, 1)).flatten()
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

    def evaluate_ihdp_ensemble(
        self,
        seed: int = 42,
        scale_outcome_for_ours: bool = False,
        model_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> Optional[Dict]:
        """Compute ensemble of model predictions on IHDP. Returns PEHE and ATE for the ensemble."""
        if model_names is None:
            model_names = ["ours", "gb-s"]
        model_names = [m.lower().replace(" ", "-").replace("_", "-") for m in model_names]
        # Map user input to baseline keys (support "gbs" -> GB-S, "ours" -> Ours, etc.)
        name_map = {
            "ours": "Ours", "gb-s": "GB-S", "gbs": "GB-S",
            "gb-t": "GB-T", "gbt": "GB-T", "tabpfn": "TabPFN",
            "transtee": "TransTEE", "ridge": "Ridge",
        }
        resolved = []
        for m in model_names:
            found = False
            for k, v in name_map.items():
                if k == m or v.lower().replace(" ", "-") == m:
                    resolved.append(v)
                    found = True
                    break
            if not found:
                for bk in self.baselines:
                    if bk.lower().replace(" ", "-").replace("_", "-") == m:
                        resolved.append(bk)
                        found = True
                        break
            if not found:
                resolved = []
                break
        if not resolved or any(r not in self.baselines for r in resolved):
            return None
        if weights is None:
            weights = [1.0 / len(resolved)] * len(resolved)
        dataset = IHDPDataset()
        x, t, y, true_cate = dataset.load()
        if x is None:
            return None
        n_samples = len(y)
        train_val_idx, test_idx = get_benchmark_indices(n_samples, seed=seed, test_frac=0.2)
        x_train, t_train, y_train = x[train_val_idx], t[train_val_idx], y[train_val_idx]
        x_test = x[test_idx]
        true_cate_test = true_cate[test_idx]
        x_train, x_test = scale_covariates(x_train, x_test)
        support_x = np.hstack([x_train, t_train.reshape(-1, 1)])
        support_y = y_train
        scaler = None
        if scale_outcome_for_ours and "Ours" in resolved:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(support_y.reshape(-1, 1))
        x_test_0 = np.hstack([x_test, np.zeros((len(x_test), 1))])
        x_test_1 = np.hstack([x_test, np.ones((len(x_test), 1))])
        query_x = np.stack([x_test_0, x_test_1], axis=0)
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(self.device).unsqueeze(0)
        support_y_scaled = scaler.transform(support_y.reshape(-1, 1)).flatten() if scaler else support_y
        t_support_x = to_tensor(support_x)
        t_support_y = to_tensor(support_y_scaled if scaler else support_y)
        t_query_x = to_tensor(query_x)
        ft = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        cards = torch.zeros(support_x.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        pred_cate_sum = None
        for i, name in enumerate(resolved):
            model = self.baselines[name]
            use_scaled = (name == "Ours" and scaler is not None)
            sy = to_tensor(support_y_scaled) if use_scaled else to_tensor(support_y)
            try:
                if name == "Ours":
                    with torch.no_grad():
                        out = self.model.model(t_support_x, sy, t_query_x, ft, cards)
                else:
                    out = model(t_support_x, sy, t_query_x, ft, cards)
                preds = out["prediction"].squeeze(0).cpu().numpy()
                pred_y0, pred_y1 = preds[0], preds[1]
                if use_scaled and scaler:
                    pred_y0 = scaler.inverse_transform(pred_y0.reshape(-1, 1)).flatten()
                    pred_y1 = scaler.inverse_transform(pred_y1.reshape(-1, 1)).flatten()
                p = (pred_y1 - pred_y0) * weights[i]
                pred_cate_sum = p if pred_cate_sum is None else pred_cate_sum + p
            except Exception:
                return None
        pred_cate = pred_cate_sum
        pehe = float(np.sqrt(np.mean((pred_cate - true_cate_test) ** 2)))
        ate_error = float(np.abs(np.mean(pred_cate) - np.mean(true_cate_test)))
        return {"PEHE": pehe, "ATE_Err": ate_error}

    def evaluate_twins(self, seed: int = 42, max_samples: Optional[int] = None) -> Dict:
        """Evaluate on Twins dataset with given random seed for train/test split."""
        print("\nEvaluating on Twins Dataset...")
        support_x, support_y, query_x_w0, query_x_w1, y0_test, y1_test, _ = load_twins(
            seed=seed, train_frac=0.8, max_samples=max_samples
        )

        true_cate_test = y1_test - y0_test
        query_x = np.stack([query_x_w0, query_x_w1], axis=0)  # [W=2, Nq, d]

        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(self.device).unsqueeze(0)
        t_support_x = to_tensor(support_x)
        t_support_y = to_tensor(support_y)
        t_query_x = to_tensor(query_x)

        results = {}
        for name, model in tqdm(self.baselines.items(), desc="Twins Models"):
            try:
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
                    outputs = model(
                        t_support_x, t_support_y, t_query_x,
                        ft.unsqueeze(0), cards.unsqueeze(0)
                    )
                    preds = outputs['prediction'].squeeze(0).cpu().numpy()

                pred_y0, pred_y1 = preds[0], preds[1]
                pred_cate = pred_y1 - pred_y0
                pehe = np.sqrt(np.mean((pred_cate - true_cate_test) ** 2))
                ate_error = np.abs(np.mean(pred_cate) - np.mean(true_cate_test))
                results[name] = {'PEHE': float(pehe), 'ATE_Err': float(ate_error)}
            except Exception as e:
                print(f"Error {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def evaluate_acic(
        self,
        data_path: str,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Evaluate on ACIC dataset. data_path: CSV or directory with x.csv + zymu_*.csv."""
        print("\nEvaluating on ACIC Dataset...")
        support_x, support_y, query_x_w0, query_x_w1, y0_test, y1_test, _, has_potential = load_acic(
            path=data_path, train_frac=0.8, seed=seed
        )
        if max_samples is not None:
            n_sup, n_q = len(support_x), len(query_x_w0)
            if n_sup + n_q > max_samples:
                n_sup_new = max(100, int(max_samples * 0.8))
                n_q_new = max(20, max_samples - n_sup_new)
                rng = np.random.RandomState(seed)
                sup_idx = rng.choice(n_sup, min(n_sup, n_sup_new), replace=False)
                q_idx = rng.choice(n_q, min(n_q, n_q_new), replace=False)
                support_x, support_y = support_x[sup_idx], support_y[sup_idx]
                query_x_w0 = query_x_w0[q_idx]
                query_x_w1 = query_x_w1[q_idx]
                y0_test, y1_test = y0_test[q_idx], y1_test[q_idx]
        if not has_potential:
            print("ACIC: No potential outcomes (mu0/mu1); using ATE only (PEHE unavailable).")

        X_train = support_x[:, :-1]
        X_test = query_x_w0[:, :-1]
        X_train, X_test = scale_covariates(X_train, X_test)
        t_train = support_x[:, -1:]
        support_x = np.hstack([X_train, t_train]).astype(np.float32)
        nq = len(X_test)
        query_x_w0 = np.hstack([X_test, np.zeros((nq, 1), dtype=np.float32)])
        query_x_w1 = np.hstack([X_test, np.ones((nq, 1), dtype=np.float32)])

        true_cate_test = y1_test - y0_test
        query_x = np.stack([query_x_w0, query_x_w1], axis=0)

        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32).to(self.device).unsqueeze(0)
        t_support_x = to_tensor(support_x)
        t_support_y = to_tensor(support_y)
        t_query_x = to_tensor(query_x)

        results = {}
        for name, model in tqdm(self.baselines.items(), desc="ACIC Models"):
            try:
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
                    outputs = model(
                        t_support_x, t_support_y, t_query_x,
                        ft.unsqueeze(0), cards.unsqueeze(0)
                    )
                    preds = outputs['prediction'].squeeze(0).cpu().numpy()

                pred_y0, pred_y1 = preds[0], preds[1]
                pred_cate = pred_y1 - pred_y0
                pehe = np.sqrt(np.mean((pred_cate - true_cate_test) ** 2)) if has_potential else float('nan')
                ate_error = np.abs(np.mean(pred_cate) - np.mean(true_cate_test))
                results[name] = {
                    'PEHE': float(pehe) if not np.isnan(pehe) else None,
                    'ATE_Err': float(ate_error)
                }
            except Exception as e:
                print(f"Error {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def print_summary(self, results, dataset_name: str = "IHDP"):
        print("\n" + "="*60)
        print(f"{dataset_name.upper()} RESULTS")
        print("="*60)
        print(f"{'Model':<20} | {'PEHE':<10} | {'ATE Err':<10}")
        print("-" * 60)
        for model, metrics in results.items():
             if 'error' in metrics:
                 print(f"{model:<20} | {'ERROR':<10} | {'ERROR':<10}")
             else:
                 pehe_str = f"{metrics['PEHE']:.4f}" if metrics.get('PEHE') is not None else "N/A"
                 print(f"{model:<20} | {pehe_str:<10} | {metrics['ATE_Err']:<10.4f}")


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to our model checkpoint (optional; if missing, only baselines run)')
    parser.add_argument('--output', type=str, default='ihdp_results.json')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda, cpu, or auto (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/test split')
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint or not Path(args.checkpoint).exists():
        print("Note: No checkpoint provided or file missing. Running baselines only (Ours skipped).")
    benchmark = RealWorldBenchmark(args.checkpoint, device=device)
    results = benchmark.evaluate_ihdp(seed=args.seed)
    benchmark.print_summary(results)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
