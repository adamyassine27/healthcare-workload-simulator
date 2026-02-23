"""
Simplified Workload Calculator
"""

import numpy as np

def calculate_utilization(num_nurses, arrival_rate, avg_service_time):
    lambda_rate = arrival_rate
    mu_rate = 1 / avg_service_time
    rho = lambda_rate / (num_nurses * mu_rate)
    utilization = min(rho * 100, 100)
    return utilization


def calculate_missed_care(utilization, shift_length_hours):
    if utilization < 70:
        base_missed = 2 + 0.05 * utilization
    elif utilization < 85:
        base_missed = -35 + 0.5 * utilization
    elif utilization < 95:
        base_missed = -100 + 1.2 * utilization
    else:
        base_missed = min(50, 0.8 * utilization)
    
    fatigue_factor = 1 + 0.05 * (shift_length_hours - 8)
    missed_care_pct = base_missed * fatigue_factor
    
    return max(0, min(100, missed_care_pct))


def calculate_wait_time(num_nurses, arrival_rate, avg_service_time):
    lambda_rate = arrival_rate / 60
    mu_rate = 1 / (avg_service_time * 60)
    c = num_nurses
    rho = lambda_rate / (c * mu_rate)
    
    if rho >= 1:
        return 999
    
    Wq = (rho / (c * mu_rate * (1 - rho)))
    wait_time_minutes = Wq
    
    return wait_time_minutes


def estimate_biomechanical_load(num_nurses, arrival_rate, shift_length):
    total_patients = arrival_rate * shift_length
    patients_per_nurse = total_patients / num_nurses
    avg_load_per_patient = 5 * 1.5
    cumulative_load = patients_per_nurse * avg_load_per_patient
    
    utilization = calculate_utilization(num_nurses, arrival_rate, 0.5)
    
    if utilization > 85:
        cumulative_load *= 1.2
    
    return cumulative_load


class WorkloadCalculator:
    
    def __init__(self):
        self.avg_service_time = 0.5
        
    def calculate_all_metrics(self, params):
        acuity_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3
        }[params['acuity_level']]
        
        service_time = self.avg_service_time * acuity_multiplier
        
        utilization = calculate_utilization(
            params['num_nurses'],
            params['arrival_rate'],
            service_time
        )
        
        missed_care = calculate_missed_care(
            utilization,
            params['shift_length']
        )
        
        wait_time = calculate_wait_time(
            params['num_nurses'],
            params['arrival_rate'],
            service_time
        )
        
        biomech_load = estimate_biomechanical_load(
            params['num_nurses'],
            params['arrival_rate'],
            params['shift_length']
        )
        
        risk_factors = []
        if utilization > 90:
            risk_factors.append("Critical utilization")
        if missed_care > 20:
            risk_factors.append("High missed care")
        if biomech_load > 80:
            risk_factors.append("Injury risk")
        if wait_time > 20:
            risk_factors.append("Long wait times")
        
        if len(risk_factors) >= 2:
            overall_risk = "High"
        elif len(risk_factors) == 1:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"
        
        return {
            'utilization': round(utilization, 1),
            'missed_care_pct': round(missed_care, 1),
            'avg_wait_time_min': round(wait_time, 1),
            'biomech_load_nh': round(biomech_load, 1),
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'patients_per_nurse': round((params['arrival_rate'] * params['shift_length']) / params['num_nurses'], 1)
        }
    
    def generate_recommendations(self, current_metrics, params):
        recommendations = []
        
        if current_metrics['utilization'] > 85:
            target_util = 75
            current_workload = params['arrival_rate'] * self.avg_service_time
            nurses_needed = int(np.ceil(current_workload / (target_util / 100)))
            additional_nurses = nurses_needed - params['num_nurses']
            
            recommendations.append({
                'priority': 'High',
                'category': 'Staffing',
                'text': f"Add {additional_nurses} nurse(s) to reduce utilization to ~{target_util}%"
            })
        
        if current_metrics['missed_care_pct'] > 15:
            recommendations.append({
                'priority': 'High',
                'category': 'Quality',
                'text': "Implement task prioritization protocol to reduce missed care"
            })
        
        if current_metrics['biomech_load_nh'] > 80:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Ergonomics',
                'text': "Schedule additional breaks or rotate tasks to reduce physical load"
            })
        
        if params['shift_length'] > 10:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Scheduling',
                'text': "Consider shorter shifts to reduce fatigue-related errors"
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'Low',
                'category': 'Status',
                'text': "✓ All metrics within acceptable ranges"
            })
        
        return recommendations