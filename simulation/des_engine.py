"""
Discre Event Nurse Workload Simulation (DES Engine)
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Task duration parameters
TASK_DURATIONS = {
    'triage': {'min': 3, 'mode': 5, 'max': 10},
    'assessment': {'mean': 12, 'std': 3},
    'vital_signs': {'min': 2, 'mode': 3, 'max': 5},
    'medication': {'min': 5, 'mode': 8, 'max': 15},
    'IV_start': {'min': 5, 'mode': 10, 'max': 20},
    'documentation': {'min': 5, 'mode': 7, 'max': 12},
    'lab_draw': {'min': 3, 'mode': 5, 'max': 8},
    'EKG': {'min': 5, 'mode': 7, 'max': 10},
}

def generate_task_duration(task_type):
    if task_type in TASK_DURATIONS:
        params = TASK_DURATIONS[task_type]
        if 'min' in params:
            return np.random.triangular(params['min'], params['mode'], params['max'])
        elif 'mean' in params:
            return max(2, np.random.normal(params['mean'], params['std']))
    return 5

def generate_patient_acuity():
    return np.random.choice([1, 2, 3, 4, 5], 
                           p=[0.13, 0.35, 0.40, 0.10, 0.02])

def calculate_required_tasks(acuity):
    tasks = ['triage', 'vital_signs', 'assessment']
    if acuity >= 3:
        tasks.extend(['medication', 'documentation'])
    if acuity >= 4:
        tasks.extend(['IV_start', 'lab_draw', 'EKG'])
    return tasks

@dataclass
class Patient:
    id: int
    arrival_time: float
    acuity: int
    env: simpy.Environment
    
    triage_start: Optional[float] = None
    triage_end: Optional[float] = None
    assessment_start: Optional[float] = None
    discharge_time: Optional[float] = None
    
    required_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[Dict] = field(default_factory=list)
    missed_tasks: List[str] = field(default_factory=list)
    
    total_wait_time: float = 0
    
    def __post_init__(self):
        self.required_tasks = calculate_required_tasks(self.acuity)
    
    def complete_task(self, task_name, start_time, duration, nurse_id):
        self.completed_tasks.append({
            'task': task_name,
            'start': start_time,
            'duration': duration,
            'nurse': nurse_id,
            'end': start_time + duration
        })
    
    def mark_task_missed(self, task_name):
        self.missed_tasks.append(task_name)
    
    def time_in_system(self):
        if self.discharge_time:
            return self.discharge_time - self.arrival_time
        return None
    
    def get_completion_rate(self):
        total_required = len(self.required_tasks)
        completed = len(self.completed_tasks)
        return completed / total_required if total_required > 0 else 0


class NurseResource:
    def __init__(self, nurse_id, env):
        self.id = nurse_id
        self.env = env
        self.current_patient = None
        self.is_busy = False
        self.total_tasks = 0
        self.total_busy_time = 0
        self.shift_start = 0
        self.task_history = []
        self.idle_periods = []
        self.last_task_end = 0
        
    def start_task(self, patient, task_name):
        self.is_busy = True
        self.current_patient = patient
        task_start = self.env.now
        return task_start
    
    def complete_task(self, patient, task_name, start_time, duration):
        self.total_tasks += 1
        self.total_busy_time += duration
        
        self.task_history.append({
            'patient_id': patient.id,
            'task': task_name,
            'start': start_time,
            'duration': duration,
            'end': self.env.now
        })
        
        patient.complete_task(task_name, start_time, duration, self.id)
        
        self.is_busy = False
        self.current_patient = None
        self.last_task_end = self.env.now
    
    def get_utilization(self):
        shift_duration = self.env.now - self.shift_start
        return self.total_busy_time / shift_duration if shift_duration > 0 else 0
    
    def get_tasks_per_hour(self):
        shift_hours = (self.env.now - self.shift_start) / 60
        return self.total_tasks / shift_hours if shift_hours > 0 else 0


class EDSimulation:
    def __init__(self, env, num_nurses, arrival_rate):
        self.env = env
        self.num_nurses = num_nurses
        self.arrival_rate = arrival_rate
        
        self.nurse_pool = simpy.Resource(env, capacity=num_nurses)
        self.nurses = [NurseResource(i, env) for i in range(num_nurses)]
        
        self.patients = []
        self.patient_counter = 0
        
        self.metrics = {
            'total_patients': 0,
            'total_tasks_completed': 0,
            'total_tasks_missed': 0,
            'average_wait_time': 0,
            'average_time_in_system': 0,
        }
    
    def patient_generator(self):
        while True:
            inter_arrival_time = np.random.exponential(60 / self.arrival_rate)
            yield self.env.timeout(inter_arrival_time)
            
            acuity = generate_patient_acuity()
            patient = Patient(
                id=self.patient_counter,
                arrival_time=self.env.now,
                acuity=acuity,
                env=self.env
            )
            
            self.patient_counter += 1
            self.patients.append(patient)
            self.metrics['total_patients'] += 1
            
            self.env.process(self.patient_pathway(patient))
    
    def patient_pathway(self, patient):
        # Triage
        with self.nurse_pool.request() as request:
            wait_start = self.env.now
            yield request
            wait_duration = self.env.now - wait_start
            patient.total_wait_time += wait_duration
            
            nurse = self._get_available_nurse()
            patient.triage_start = self.env.now
            task_duration = generate_task_duration('triage')
            nurse.start_task(patient, 'triage')
            yield self.env.timeout(task_duration)
            nurse.complete_task(patient, 'triage', patient.triage_start, task_duration)
            patient.triage_end = self.env.now
        
        # Other tasks
        for task in patient.required_tasks:
            if task == 'triage':
                continue
            
            if self._should_miss_task(nurse):
                patient.mark_task_missed(task)
                self.metrics['total_tasks_missed'] += 1
                continue
            
            with self.nurse_pool.request() as request:
                wait_start = self.env.now
                yield request
                wait_duration = self.env.now - wait_start
                patient.total_wait_time += wait_duration
                
                nurse = self._get_available_nurse()
                task_start = self.env.now
                task_duration = generate_task_duration(task)
                
                nurse.start_task(patient, task)
                yield self.env.timeout(task_duration)
                nurse.complete_task(patient, task, task_start, task_duration)
                
                self.metrics['total_tasks_completed'] += 1
        
        patient.discharge_time = self.env.now
    
    def _get_available_nurse(self):
        for nurse in self.nurses:
            if not nurse.is_busy:
                return nurse
        return self.nurses[0]
    
    def _should_miss_task(self, nurse):
        utilization = nurse.get_utilization()
        if utilization > 0.90:
            return np.random.random() < 0.30
        elif utilization > 0.85:
            return np.random.random() < 0.15
        return False
    
    #Run simulation and calculate final metrics
    def run(self, sim_time=1440):
        self.env.process(self.patient_generator())
        self.env.run(until=sim_time)
    
    def calculate_final_metrics(self):
        completed_patients = [p for p in self.patients if p.discharge_time]
        
        if completed_patients:
            times_in_system = [p.time_in_system() for p in completed_patients]
            wait_times = [p.total_wait_time for p in completed_patients]
            
            self.metrics['average_time_in_system'] = np.mean(times_in_system)
            self.metrics['average_wait_time'] = np.mean(wait_times)
        
        self.metrics['nurse_utilizations'] = [n.get_utilization() for n in self.nurses]
        self.metrics['nurse_tasks_per_hour'] = [n.get_tasks_per_hour() for n in self.nurses]


def run_single_scenario(num_nurses, arrival_rate, sim_time=1440):
    env = simpy.Environment()
    sim = EDSimulation(env, num_nurses, arrival_rate)
    sim.run(sim_time)
    sim.calculate_final_metrics()
    return sim