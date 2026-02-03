"""Basic usage example."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from inference.api import ParallelUniverseModel, Intervention


def generate_synthetic_data(n_samples=100):
    """Generate synthetic tabular data for demo."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.uniform(18, 80, n_samples),
        'income': np.random.uniform(20000, 150000, n_samples),
        'education': np.random.randint(0, 5, n_samples),  # 0-4 levels
        'experience': np.random.uniform(0, 40, n_samples),
    })
    
    return data


def main():
    print("Parallel Universe Transformer - Basic Usage Example")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(200)
    
    # Split into support and query
    support_data = data[:100]
    query_data = data[100:110]  # Small query set for demo
    
    print(f"   Support set: {len(support_data)} samples")
    print(f"   Query set: {len(query_data)} samples")
    
    # Define interventions
    print("\n2. Defining interventions...")
    interventions = [
        Intervention(feature='age', type='set', value=30),
        Intervention(feature='income', type='shift', value=10000),
        Intervention(feature='education', type='set', value=4),
    ]
    
    for i, intv in enumerate(interventions):
        print(f"   {i+1}. {intv.type.upper()} {intv.feature} to {intv.value}")
    
    # Load model (you need a trained checkpoint)
    print("\n3. Loading model...")
    try:
        model = ParallelUniverseModel.from_pretrained('checkpoints/final_model.pt')
        print("   Model loaded successfully!")
        
        # Predict
        print("\n4. Predicting counterfactuals...")
        results = model.predict_interventions(
            data=support_data,
            query=query_data,
            interventions=interventions
        )
        
        # Display results
        print("\n5. Results:")
        print(f"   Baseline predictions: {results.baseline[:5]}")
        print(f"   Counterfactual predictions shape: {results.counterfactuals.shape}")
        print(f"   Deltas (treatment effects) shape: {results.deltas.shape}")
        print(f"   Uncertainty estimates shape: {results.uncertainty.shape}")
        
        # Show effect of first intervention
        print(f"\n6. Effect of first intervention ({interventions[0].feature} = {interventions[0].value}):")
        for i in range(min(5, len(query_data))):
            baseline = results.baseline[i]
            cf = results.counterfactuals[0, i]
            delta = results.deltas[0, i]
            uncertainty = results.uncertainty[1, i]  # Intervention 0 (index 1 includes baseline)
            
            print(f"   Sample {i}: baseline={baseline:.3f}, counterfactual={cf:.3f}, "
                  f"delta={delta:.3f} ± {uncertainty:.3f}")
        
    except FileNotFoundError:
        print("   Error: Model checkpoint not found!")
        print("   Please train a model first using: python train_model.py")
        print("\n   For this demo, we'll skip the prediction step.")
    
    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == '__main__':
    main()
