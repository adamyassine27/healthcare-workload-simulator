"""
Feature Engineering for ML Models
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    Input columns expected from DES:
    - num_nurses
    - arrival_rate  
    - sim_hours
    - num_patients
    
    Output: Original + derived features
    """
    df = df.copy()
    
    # Basic ratios
    df['nurse_patient_ratio'] = df['num_nurses'] / df['num_patients']
    df['patients_per_nurse'] = df['num_patients'] / df['num_nurses']
    
    # Workload metrics
    df['total_workload'] = df['arrival_rate'] * df['sim_hours']
    df['workload_per_nurse'] = df['total_workload'] / df['num_nurses']
    df['workload_intensity'] = df['total_workload'] / (df['num_nurses'] * df['sim_hours'])
    
    # Theoretical predictions
    df['theoretical_util'] = (df['arrival_rate'] * 0.5) / df['num_nurses'] * 100
    
    # Interaction features
    df['nurses_x_arrival'] = df['num_nurses'] * df['arrival_rate']
    df['hours_x_arrival'] = df['sim_hours'] * df['arrival_rate']
    df['nurses_x_hours'] = df['num_nurses'] * df['sim_hours']
    
    return df


def prepare_ml_data(df: pd.DataFrame):
    """
    Prepare features (X) and targets (y) for ML training
    
    Returns:
    - X: Feature matrix
    - targets: Dictionary of target variables
    - feature_cols: List of feature column names
    """
    # Engineer features
    df = engineer_features(df)
    
        # Define feature columns (original + derived)
    feature_cols = [
        # Original parameters
        'num_nurses',
        'arrival_rate',
        'sim_hours',
        'num_patients',
        
        # Derived features
        'nurse_patient_ratio',
        'patients_per_nurse',
        'total_workload',
        'workload_per_nurse',
        'workload_intensity',
        'theoretical_util',
        
        # Interactions
        'nurses_x_arrival',
        'hours_x_arrival',
        'nurses_x_hours',
    ]
    
    X = df[feature_cols]
    
    # Define targets based on DES outputs
    targets = {
        'utilization': df['avg_utilization'],
        'missed_care': df['missed_care_pct'],
        'wait_time': df['avg_wait_time'],
        'tasks_per_hour': df['avg_tasks_per_hour'],
    }
    
    return X, targets, feature_cols