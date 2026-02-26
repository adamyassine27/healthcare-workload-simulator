#Generate Training Data from DES Simulations

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from scipy.stats import qmc
from simulation.des_engine import run_single_scenario

def generate_training_dataset(num_scenarios=100000, reps_per_scenario=10):
    """
    Generate comprehensive training dataset
    
    Parameters:
    - num_scenarios: Number of unique parameter combinations
    - reps_per_scenario: Monte Carlo replications for each scenario
    
    Returns:
    - DataFrame with all scenarios and results
    """
    print("="*70)
    print("GENERATING TRAINING DATA FROM DES SIMULATIONS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Replications per scenario: {reps_per_scenario}")
    print(f"  Total simulations: {num_scenarios * reps_per_scenario}")
    print(f"  Estimated time: {num_scenarios * reps_per_scenario * 0.5 / 60:.1f} minutes\n")
    
    # Latin Hypercube Sampling for parameter space coverage
    print("Generating parameter combinations using Latin Hypercube Sampling...")
    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=num_scenarios)
    
    # Define parameter ranges matching DES
    param_combinations = []
    for s in sample:
        nurses = int(np.interp(s[0], [0, 1], [2, 20]))  # 2-20 nurses
        arrival = np.interp(s[1], [0, 1], [1.0, 20.0])  # 1-20 patients/hour
        sim_hours = int(np.interp(s[2], [0, 1], [6, 16]))  # 6-16 hours
        
        param_combinations.append({
            'num_nurses': nurses,
            'arrival_rate': round(arrival, 1),
            'sim_hours': sim_hours
        })
    
    print(f"✓ Generated {len(param_combinations)} unique scenarios\n")
    
    # Run simulations
    print("Running DES simulations...")
    print("-" * 70)
    
    all_results = []
    
    for idx, params in enumerate(param_combinations):
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{num_scenarios} scenarios ({(idx+1)/num_scenarios*100:.1f}%)")
        
        scenario_results = []
        
        # Run multiple replications for this scenario
        for rep in range(reps_per_scenario):
            # Run DES
            sim = run_single_scenario(
                num_nurses=params['num_nurses'],
                arrival_rate=params['arrival_rate'],
                sim_time=params['sim_hours'] * 60  # Convert hours to minutes
            )
            
            # Extract metrics
            total_tasks = sim.metrics['total_tasks_completed'] + sim.metrics['total_tasks_missed']
            
            scenario_results.append({
                'avg_utilization': np.mean(sim.metrics['nurse_utilizations']) * 100,
                'avg_tasks_per_hour': np.mean(sim.metrics['nurse_tasks_per_hour']),
                'missed_care_pct': (sim.metrics['total_tasks_missed'] / total_tasks * 100
                                   if total_tasks > 0 else 0),
                'avg_wait_time': sim.metrics['average_wait_time'],
                'total_patients': sim.metrics['total_patients'],
            })
        
        # Aggregate replications (mean and std)
        aggregated = {
            'scenario_id': idx,
            'num_nurses': params['num_nurses'],
            'arrival_rate': params['arrival_rate'],
            'sim_hours': params['sim_hours'],
            
            # Patient metrics
            'num_patients': np.mean([r['total_patients'] for r in scenario_results]),
            
            # Utilization
            'avg_utilization': np.mean([r['avg_utilization'] for r in scenario_results]),
            'utilization_std': np.std([r['avg_utilization'] for r in scenario_results]),
            
            # Missed care
            'missed_care_pct': np.mean([r['missed_care_pct'] for r in scenario_results]),
            'missed_care_std': np.std([r['missed_care_pct'] for r in scenario_results]),
            
            # Wait time
            'avg_wait_time': np.mean([r['avg_wait_time'] for r in scenario_results]),
            'wait_time_std': np.std([r['avg_wait_time'] for r in scenario_results]),
            
            # Tasks per hour
            'avg_tasks_per_hour': np.mean([r['avg_tasks_per_hour'] for r in scenario_results]),
            'tasks_per_hour_std': np.std([r['avg_tasks_per_hour'] for r in scenario_results]),
        }
        
        all_results.append(aggregated)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = '../data/training_data.csv'
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("✓ TRAINING DATA GENERATION COMPLETE")
    print("="*70)
    print(f"  Output file: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Total simulations run: {num_scenarios * reps_per_scenario}")
    
    # Show summary statistics
    print("\nDataset Summary:")
    print("-" * 70)
    print(df[['num_nurses', 'arrival_rate', 'sim_hours', 
             'avg_utilization', 'missed_care_pct']].describe())
    
    return df


if __name__ == "__main__":
    print("\n" + " "*20)
    print("ED WORKLOAD SIMULATION - TRAINING DATA GENERATION")
    print(" "*20 + "\n")
    
    training_data = generate_training_dataset(
        num_scenarios=100000,
        reps_per_scenario=10
    )
    
    print("\n✓ Complete! You can now run 02_train_ml_models.py\n")