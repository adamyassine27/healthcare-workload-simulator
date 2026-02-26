"""
Train ML Models on DES-Generated Data
"""

import sys
sys.path.append('..')

from ml_models.training import WorkloadMLModels
from ml_models.feature_engineering import prepare_ml_data
import pandas as pd

def main():
    print("="*70)
    print("STEP 2: TRAINING ML MODELS")
    print("="*70)
    
    # Load training data
    print("\n1. Loading training data...")
    df = pd.read_csv('../data/training_data.csv')
    print(f"   ✓ Loaded {len(df)} scenarios")
    
    # Prepare features and targets
    print("\n2. Preparing features...")
    X, targets, feature_names = prepare_ml_data(df)
    print(f"   ✓ Features: {len(feature_names)}")
    print(f"   ✓ Samples: {len(X)}")
    print(f"   ✓ Targets: {list(targets.keys())}")
    
    # Train models
    print("\n3. Training models...")
    models = WorkloadMLModels()
    models.train(X, targets)
    
    # Save models
    print("\n4. Saving models...")
    models.save('../ml_models/models.pkl')
    
    # Display final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nModel Performance:")
    for name, stats in models.training_stats.items():
        print(f"  {name:15s} → R²={stats['test_r2']:.3f}, RMSE={stats['test_rmse']:.2f}")
    
    print("\n✓ Models saved to: ml_models/models.pkl")
    print("✓ You can now run: 03_validate_models.py\n")

if __name__ == "__main__":
    main()