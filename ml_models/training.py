"""
ML Model Training Pipeline
Trains Random Forest models on DES simulation data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from .feature_engineering import prepare_ml_data


class WorkloadMLModels:
    """
    Container for all ML meta-models
    One model per target variable
    """
    
    def __init__(self):
        self.model_utilization = None
        self.model_missed_care = None
        self.model_wait_time = None
        self.model_tasks_per_hour = None
        
        self.feature_names = None
        self.training_stats = {}
    
    def train(self, X, targets, test_size=0.2, random_state=42):
        """
        Train all models
        
        Parameters:
        - X: Feature matrix (DataFrame)
        - targets: Dict of target variables
        - test_size: Fraction for test set
        - random_state: Random seed
        """
        print("="*70)
        print("TRAINING ML META-MODELS")
        print("="*70)
        
        self.feature_names = X.columns.tolist()
        print(f"\nFeatures: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        
        # Split data once for all models
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        
        # Train each model
        models_config = [
            ('Utilization', 'utilization', 'model_utilization'),
            ('Missed Care', 'missed_care', 'model_missed_care'),
            ('Wait Time', 'wait_time', 'model_wait_time'),
            ('Tasks/Hour', 'tasks_per_hour', 'model_tasks_per_hour'),
        ]
        
        for idx, (name, target_key, attr_name) in enumerate(models_config, 1):
            print(f"\n{idx}. Training {name} Model...")
            print("-" * 50)
            
            # Split target
            y_train, y_test = train_test_split(
                targets[target_key], 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Initialize model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='r2', n_jobs=-1
            )
            
            # Store statistics
            self.training_stats[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            # Print results
            print(f"  Train R²:  {train_r2:.4f}")
            print(f"  Test R²:   {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test MAE:  {test_mae:.2f}")
            print(f"  CV R²:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Store model
            setattr(self, attr_name, model)
        
        print("\n" + "="*70)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print("\nModel Performance Summary:")
        print("-" * 50)
        for name, stats in self.training_stats.items():
            print(f"{name:15s} | R²={stats['test_r2']:.3f} | RMSE={stats['test_rmse']:.2f}")
    
    def predict(self, X):
        """
        Make predictions with uncertainty quantification
        
        Returns dict with mean, std, and confidence intervals
        """
        # Get predictions from all trees
        util_preds = np.array([tree.predict(X) for tree in self.model_utilization.estimators_])
        missed_preds = np.array([tree.predict(X) for tree in self.model_missed_care.estimators_])
        wait_preds = np.array([tree.predict(X) for tree in self.model_wait_time.estimators_])
        tasks_preds = np.array([tree.predict(X) for tree in self.model_tasks_per_hour.estimators_])
        
        return {
            'utilization': {
                'mean': util_preds.mean(axis=0),
                'std': util_preds.std(axis=0),
                'ci_lower': np.percentile(util_preds, 2.5, axis=0),
                'ci_upper': np.percentile(util_preds, 97.5, axis=0),
            },
            'missed_care': {
                'mean': missed_preds.mean(axis=0),
                'std': missed_preds.std(axis=0),
                'ci_lower': np.percentile(missed_preds, 2.5, axis=0),
                'ci_upper': np.percentile(missed_preds, 97.5, axis=0),
            },
            'wait_time': {
                'mean': wait_preds.mean(axis=0),
                'std': wait_preds.std(axis=0),
                'ci_lower': np.percentile(wait_preds, 2.5, axis=0),
                'ci_upper': np.percentile(wait_preds, 97.5, axis=0),
            },
            'tasks_per_hour': {
                'mean': tasks_preds.mean(axis=0),
                'std': tasks_preds.std(axis=0),
                'ci_lower': np.percentile(tasks_preds, 2.5, axis=0),
                'ci_upper': np.percentile(tasks_preds, 97.5, axis=0),
            },
        }
    
    def save(self, filepath='ml_models/models.pkl'):
        """Save models to disk"""
        joblib.dump({
            'utilization': self.model_utilization,
            'missed_care': self.model_missed_care,
            'wait_time': self.model_wait_time,
            'tasks_per_hour': self.model_tasks_per_hour,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }, filepath)
        print(f"\n✓ Models saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='ml_models/models.pkl'):
        """Load models from disk"""
        obj = cls()
        data = joblib.load(filepath)
        obj.model_utilization = data['utilization']
        obj.model_missed_care = data['missed_care']
        obj.model_wait_time = data['wait_time']
        obj.model_tasks_per_hour = data['tasks_per_hour']
        obj.feature_names = data['feature_names']
        obj.training_stats = data['training_stats']
        return obj