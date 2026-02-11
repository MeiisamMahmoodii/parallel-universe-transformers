"""Run ablation experiments."""

import os
import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_root / "code"))

from train.config import TrainingConfig
from train.trainer import Trainer
from model.model import ParallelUniverseTransformer


def run_cross_attention_ablation():
    """Ablation: Impact of cross-world attention."""
    print("\n" + "="*60)
    print("ABLATION: Cross-World Attention")
    print("="*60 + "\n")
    
    configs = [
        {
            'name': 'no_cross_attention',
            'cross_world_layers': [],
            'description': 'No cross-world attention (independent worlds)'
        },
        {
            'name': 'baseline_only',
            'cross_world_layers': [3, 5],
            'attend_to_all_worlds': False,
            'description': 'Cross-attention only to baseline world'
        },
        {
            'name': 'all_worlds',
            'cross_world_layers': [3, 5],
            'attend_to_all_worlds': True,
            'description': 'Cross-attention to all worlds (full model)'
        }
    ]
    
    results = {}
    
    for config_spec in configs:
        print(f"\nTraining: {config_spec['description']}")
        
        config = TrainingConfig(
            cross_world_layers=config_spec.get('cross_world_layers', [3, 5]),
            attend_to_all_worlds=config_spec.get('attend_to_all_worlds', True),
            max_steps=20000,  # Shorter for ablation
            checkpoint_dir=f"checkpoints/ablation_cross_attn_{config_spec['name']}",
            use_wandb=False
        )
        
        trainer = Trainer(config)
        trainer.train()
        
        # Store final metrics
        metrics = trainer.metrics_computer.compute()
        results[config_spec['name']] = {
            'config': config_spec,
            'metrics': metrics
        }
        
        print(f"\nResults for {config_spec['name']}:")
        print(json.dumps(metrics, indent=2))
    
    # Save results
    with open('ablation_cross_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Cross-attention ablation completed!")
    print("="*60)


def run_delta_loss_ablation():
    """Ablation: Impact of delta consistency loss."""
    print("\n" + "="*60)
    print("ABLATION: Delta Loss")
    print("="*60 + "\n")
    
    configs = [
        {
            'name': 'no_delta_loss',
            'lambda_delta': 0.0,
            'description': 'Prediction loss only (no delta loss)'
        },
        {
            'name': 'with_delta_loss',
            'lambda_delta': 1.0,
            'description': 'With delta consistency loss (full model)'
        }
    ]
    
    results = {}
    
    for config_spec in configs:
        print(f"\nTraining: {config_spec['description']}")
        
        config = TrainingConfig(
            lambda_delta=config_spec['lambda_delta'],
            max_steps=20000,
            checkpoint_dir=f"checkpoints/ablation_delta_{config_spec['name']}",
            use_wandb=False
        )
        
        trainer = Trainer(config)
        trainer.train()
        
        metrics = trainer.metrics_computer.compute()
        results[config_spec['name']] = {
            'config': config_spec,
            'metrics': metrics
        }
        
        print(f"\nResults for {config_spec['name']}:")
        print(json.dumps(metrics, indent=2))
    
    with open('ablation_delta_loss_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Delta loss ablation completed!")
    print("="*60)


def main():
    """Run all ablations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        '--ablation',
        choices=['cross_attention', 'delta_loss', 'all'],
        default='all',
        help='Which ablation to run'
    )

    args = parser.parse_args()

    if args.ablation in ['cross_attention', 'all']:
        run_cross_attention_ablation()

    if args.ablation in ['delta_loss', 'all']:
        run_delta_loss_ablation()


if __name__ == '__main__':
    main()
