"""
Validate ML Models Against New DES Simulations
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from simulation.des_engine import run_single_scenario
from ml_models.training import WorkloadMLModels
from ml_models.feature_engineering import engineer_features

def main():
    print("="*70)
    print("STEP 3: MODEL VALIDATION")
    print("="*70)
    
    # Load models
    print("\n1. Loading ML models...")
    models = WorkloadMLModels.load('../ml_models/models.pkl')
    print("   ✓ Models loaded successfully")
    
    # Define validation scenarios (different from training)
    print("\n2. Generating validation scenarios...")
    validation_scenarios = [
        {'num_nurses': 3, 'arrival_rate': 5.5, 'sim_hours': 12},
        {'num_nurses': 5, 'arrival_rate': 7.0, 'sim_hours': 16},
        {'num_nurses': 4, 'arrival_rate': 4.5, 'sim_hours': 8},
        {'num_nurses': 6, 'arrival_rate': 8.0, 'sim_hours': 12},
        {'num_nurses': 2, 'arrival_rate': 3.5, 'sim_hours': 10},
    ]
    print(f"   ✓ {len(validation_scenarios)} scenarios to validate")
    
    # Run validation
    print("\n3. Running validation (this takes ~30 minutes)...")
    print("-" * 70)
    
    results = []
    
    for idx, scenario in enumerate(validation_scenarios, 1):
        print(f"\nScenario {idx}/{len(validation_scenarios)}: {scenario}")
        
        # Run DES simulations (50 replications)
        print("   Running DES simulations (50 reps)...", end=" ")
        des_results = []
        for rep in range(50):
            sim = run_single_scenario(
                num_nurses=scenario['num_nurses'],
                arrival_rate=scenario['arrival_rate'],
                sim_time=scenario['sim_hours'] * 60
            )
            
            total_tasks = sim.metrics['total_tasks_completed'] + sim.metrics['total_tasks_missed']
            
            des_results.append({
                'utilization': np.mean(sim.metrics['nurse_utilizations']) * 100,
                'missed_care': (sim.metrics['total_tasks_missed'] / total_tasks * 100
                               if total_tasks > 0 else 0),
                'wait_time': sim.metrics['average_wait_time'],
                'total_patients': sim.metrics['total_patients'],
            })
        
        print("Done!")
        
        # Calculate DES averages
        des_avg = {
            'utilization': np.mean([r['utilization'] for r in des_results]),
            'missed_care': np.mean([r['missed_care'] for r in des_results]),
            'wait_time': np.mean([r['wait_time'] for r in des_results]),
        }
        
        # Get ML predictions
        print("   Getting ML predictions...", end=" ")
        input_df = pd.DataFrame([{
            'num_nurses': scenario['num_nurses'],
            'arrival_rate': scenario['arrival_rate'],
            'sim_hours': scenario['sim_hours'],
            'num_patients': np.mean([r['total_patients'] for r in des_results]),
        }])
        input_df = engineer_features(input_df)
        X = input_df[models.feature_names]
        ml_preds = models.predict(X)
        print("Done!")
        
        ml_avg = {
            'utilization': ml_preds['utilization']['mean'][0],
            'missed_care': ml_preds['missed_care']['mean'][0],
            'wait_time': ml_preds['wait_time']['mean'][0],
        }
        
        # Calculate errors
        errors = {
            'utilization': abs(des_avg['utilization'] - ml_avg['utilization']),
            'missed_care': abs(des_avg['missed_care'] - ml_avg['missed_care']),
            'wait_time': abs(des_avg['wait_time'] - ml_avg['wait_time']),
        }
        
        # Display comparison
        print(f"\n   Results:")
        print(f"     Utilization:  DES={des_avg['utilization']:5.1f}%  ML={ml_avg['utilization']:5.1f}%  Error={errors['utilization']:.1f}%")
        print(f"     Missed Care:  DES={des_avg['missed_care']:5.1f}%  ML={ml_avg['missed_care']:5.1f}%  Error={errors['missed_care']:.1f}%")
        print(f"     Wait Time:    DES={des_avg['wait_time']:5.1f}m  ML={ml_avg['wait_time']:5.1f}m  Error={errors['wait_time']:.1f}m")
        
        results.append({
            'scenario': scenario,
            'des': des_avg,
            'ml': ml_avg,
            'errors': errors
        })
    
    # Overall summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    avg_errors = {
        'utilization': np.mean([r['errors']['utilization'] for r in results]),
        'missed_care': np.mean([r['errors']['missed_care'] for r in results]),
        'wait_time': np.mean([r['errors']['wait_time'] for r in results]),
    }
    
    print("\nAverage Absolute Errors:")
    for metric, error in avg_errors.items():
        status = "✓ PASS" if error < 5 else "X REVIEW"
        print(f"  {metric:15s}: {error:5.2f}  {status}")
    
    # Final verdict
    all_pass = all(e < 5 for e in avg_errors.values())
    
    print("\n" + "="*70)
    if all_pass:
        print("✓ VALIDATION PASSED - All errors within 5% threshold")
        print("✓ ML models are accurate enough for deployment")
    else:
        print("X  VALIDATION WARNING - Some errors exceed 5%")
        print("   Consider retraining with more scenarios or reviewing model")
    print("="*70)
    
    print("\n✓ Validation complete!")
    print("✓ You can now integrate ML models into dashboard\n")

if __name__ == "__main__":
    main()