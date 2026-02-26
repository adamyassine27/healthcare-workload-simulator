"""
Healthcare Workload Simulator Dashboard
Streamlit dashboard + ML Meta-Model Integration
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import time
import os

#Import calculator functions
from calculator import (
    calculate_utilization,
    calculate_missed_care,
    calculate_wait_time,
    estimate_biomechanical_load,
    WorkloadCalculator
)

# Page config
st.set_page_config(
    page_title="Healthcare Workload Simulator",
    page_icon="",
    layout="wide"
)

# Import ML models
sys.path.append('..')
try:
    from ml_models.training import WorkloadMLModels
    from ml_models.feature_engineering import engineer_features
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# Import DES for verification
try:
    from simulation.des_engine import run_single_scenario
    import simpy
    DES_AVAILABLE = True
except:
    DES_AVAILABLE = False

# Load ML models
@st.cache_resource
def load_ml_models():
    if not ML_AVAILABLE:
        return None

    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'models.pkl')
    model_path = os.path.abspath(model_path)

    def is_lfs_pointer(path):
        try:
            with open(path, 'rb') as f:
                return f.read(50).startswith(b'version https://git-lfs')
        except:
            return True

    # Try loading existing real model first
    if os.path.exists(model_path) and not is_lfs_pointer(model_path):
        try:
            return WorkloadMLModels.load(model_path)
        except Exception as e:
            st.warning(f"Could not load model: {e}")

    # Retrain from scratch
    try:
        from ml_models.feature_engineering import prepare_ml_data 

        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv')
        data_path = os.path.abspath(data_path)

        df = pd.read_csv(data_path)
        X, targets, _ = prepare_ml_data(df) 

        ml = WorkloadMLModels()
        ml.train(X, targets)
        ml.save(model_path)
        return ml
    except Exception as e:
        st.error(f"ML model training failed: {e}")
        return None

try:
    ml_models = load_ml_models()
except Exception as e:
    st.error(f"ML load error: {e}")
    ml_models = None

# Title
st.title("Healthcare Workload Simulator")

# Add subtitle about method
if ml_models:
    st.markdown("**Hybrid DES-ML Framework** | Instant ML predictions with DES verification")
else:
    st.markdown("**Simplified Calculator** | Based on Neumann et al. Research")

# Sidebar inputs
st.sidebar.header("Scenario Parameters")

num_nurses = st.sidebar.slider(
    "Number of Nurses",
    min_value=2,
    max_value=20,
    value=10,
    step=1
)

arrival_rate = st.sidebar.slider(
    "Patient Arrival Rate (per hour)",
    min_value=1.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

shift_length = st.sidebar.select_slider(
    "Shift Length (hours)",
    options=[6, 8, 10, 12, 14, 16],
    value=8
)

acuity_level = st.sidebar.selectbox(
    "Patient Acuity",
    options=['low', 'medium', 'high'],
    index=1
)

# Prediction method selector
st.sidebar.markdown("---")
st.sidebar.markdown("### Prediction Method")

if ml_models and ML_AVAILABLE:
    use_ml = st.sidebar.radio(
        "Choose method:",
        options=["ML Meta-Model (Instant)", "Calculator (Formula-based)"],
        index=0
    )
    use_ml = (use_ml == "ML Meta-Model (Instant)")
else:
    use_ml = False
    st.sidebar.info("ML models not available. Using calculator.")

# Calculate predictions
params_1 = {
    'num_nurses': num_nurses,
    'arrival_rate': arrival_rate,
    'shift_length': shift_length,
    'acuity_level': acuity_level
}

if use_ml and ml_models:
    # ML PREDICTIONS
    st.markdown("## ML Predictions")
    st.caption("Trained on 500+ DES scenarios | Predictions in <1ms")
    
    start_time = time.time()
    
    # Prepare input for ML
    # Estimate number of patients
    estimated_patients = arrival_rate * shift_length
    
    input_df = pd.DataFrame([{
        'num_nurses': num_nurses,
        'arrival_rate': arrival_rate,
        'sim_hours': shift_length,
        'num_patients': estimated_patients,
    }])
    input_df = engineer_features(input_df)
    X = input_df[ml_models.feature_names]
    
    # Get predictions
    predictions = ml_models.predict(X)
    
    ml_time = (time.time() - start_time) * 1000  # milliseconds
    
    # Extract values
    metrics_1 = {
        'utilization': predictions['utilization']['mean'][0],
        'missed_care_pct': predictions['missed_care']['mean'][0],
        'avg_wait_time_min': predictions['wait_time']['mean'][0],
        'biomech_load_nh': estimate_biomechanical_load(num_nurses, arrival_rate, shift_length),
    }
    
    # Add uncertainty info
    util_ci = (predictions['utilization']['ci_lower'][0], predictions['utilization']['ci_upper'][0])
    missed_ci = (predictions['missed_care']['ci_lower'][0], predictions['missed_care']['ci_upper'][0])
    
else:
    # Calculator Predictions
    st.markdown("## Calculator Predictions")
    st.caption("Formula-based estimates using queue theory")
    
    calculator = WorkloadCalculator()
    metrics_1 = calculator.calculate_all_metrics(params_1)

# Risk assessment
risk_factors = []
if metrics_1['utilization'] > 90:
    risk_factors.append("Critical utilization")
if metrics_1['missed_care_pct'] > 20:
    risk_factors.append("High missed care")
if metrics_1.get('biomech_load_nh', 0) > 80:
    risk_factors.append("Injury risk")

if len(risk_factors) >= 2:
    overall_risk = "High"
elif len(risk_factors) == 1:
    overall_risk = "Medium"
else:
    overall_risk = "Low"

metrics_1['overall_risk'] = overall_risk
metrics_1['risk_factors'] = risk_factors

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta = None
    if use_ml and ml_models:
        delta = f"±{(util_ci[1]-util_ci[0])/2:.1f}%"
    
    st.metric(
        label="Nurse Utilization",
        value=f"{metrics_1['utilization']:.1f}%",
        delta=delta
    )

with col2:
    delta = None
    if use_ml and ml_models:
        delta = f"±{(missed_ci[1]-missed_ci[0])/2:.1f}%"
    
    st.metric(
        label="Missed Care",
        value=f"{metrics_1['missed_care_pct']:.1f}%",
        delta=delta
    )

with col3:
    st.metric(
        label="Avg Wait Time",
        value=f"{metrics_1['avg_wait_time_min']:.1f} min"
    )

with col4:
    st.metric(
        label="Biomech Load",
        value=f"{metrics_1.get('biomech_load_nh', 0):.0f} N·h"
    )

# Risk indicator
risk_colors = {
    'Low': 'green',
    'Medium': 'orange',
    'High': 'red'
}

st.markdown(f"""
<div style='padding: 10px; background-color: {risk_colors[overall_risk]}20; 
     border-left: 5px solid {risk_colors[overall_risk]}; border-radius: 5px; margin-top: 20px;'>
    <h3 style='margin: 0; color: {risk_colors[overall_risk]};'>
        Overall Risk: {overall_risk}
    </h3>
    {f"<p style='margin: 5px 0 0 0; color: #666;'>Factors: {', '.join(risk_factors)}</p>" if risk_factors else ""}
</div>
""", unsafe_allow_html=True)

# DES Verification
if DES_AVAILABLE:
    st.markdown("---")
    st.markdown("## Full DES Simulation Verification")
    
    if use_ml:
        st.markdown("Run complete DES to verify ML predictions (~30 seconds)")
    else:
        st.markdown("Run complete DES to compare with calculator estimates (~30 seconds)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_reps = st.slider("Number of replications", 5, 50, 10, 5)
    
    with col2:
        run_des = st.button("▶ Run DES", type="primary", use_container_width=True)
    
    if run_des:
        with st.spinner(f"Running DES simulation ({num_reps} replications)..."):
            
            start_time = time.time()
            
            des_results = []
            for rep in range(num_reps):
                sim = run_single_scenario(
                    num_nurses=num_nurses,
                    arrival_rate=arrival_rate,
                    sim_time=shift_length * 60
                )
                
                total_tasks = sim.metrics['total_tasks_completed'] + sim.metrics['total_tasks_missed']
                
                des_results.append({
                    'utilization': np.mean(sim.metrics['nurse_utilizations']) * 100,
                    'missed_care': (sim.metrics['total_tasks_missed'] / total_tasks * 100
                                   if total_tasks > 0 else 0),
                    'wait_time': sim.metrics['average_wait_time'],
                })
            
            des_time = time.time() - start_time
            
            # Calculate averages
            des_util = np.mean([r['utilization'] for r in des_results])
            des_util_std = np.std([r['utilization'] for r in des_results])
            des_missed = np.mean([r['missed_care'] for r in des_results])
            des_missed_std = np.std([r['missed_care'] for r in des_results])
            des_wait = np.mean([r['wait_time'] for r in des_results])
            des_wait_std = np.std([r['wait_time'] for r in des_results])
        
        st.success(f"✓ Simulation completed in {des_time:.1f}s ({num_reps} replications)")
        
        # Comparison table
        st.markdown("### Comparison Results")
        
        method_name = "ML Prediction" if use_ml else "Calculator"
        
        comparison_df = pd.DataFrame({
            'Metric': ['Utilization (%)', 'Missed Care (%)', 'Wait Time (min)'],
            method_name: [
                f"{metrics_1['utilization']:.1f}",
                f"{metrics_1['missed_care_pct']:.1f}",
                f"{metrics_1['avg_wait_time_min']:.1f}"
            ],
            'DES Simulation': [
                f"{des_util:.1f} ± {des_util_std:.1f}",
                f"{des_missed:.1f} ± {des_missed_std:.1f}",
                f"{des_wait:.1f} ± {des_wait_std:.1f}"
            ],
            'Difference': [
                f"{abs(metrics_1['utilization'] - des_util):.1f}",
                f"{abs(metrics_1['missed_care_pct'] - des_missed):.1f}",
                f"{abs(metrics_1['avg_wait_time_min'] - des_wait):.1f}"
            ],
            'Within 5%?': [
                '✓' if abs(metrics_1['utilization'] - des_util) < 5 else 'X',
                '✓' if abs(metrics_1['missed_care_pct'] - des_missed) < 5 else 'X',
                '✓' if abs(metrics_1['avg_wait_time_min'] - des_wait) < 5 else 'X',
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Validation message
        all_within_5 = all('✓' in x for x in comparison_df['Within 5%?'])
        
        if use_ml:
            if all_within_5:
                st.success("✓ ML predictions validated! All metrics within 5% of DES simulation.")
            else:
                st.info("ℹ: Some predictions differ by >5%. This may indicate an edge case scenario.")
            
            # Speed comparison
            st.metric(
                "ML Speedup",
                f"{des_time/ml_time*1000:.0f}x faster",
                f"DES: {des_time:.1f}s vs ML: {ml_time/1000:.3f}s"
            )

# Graphs
st.header("Visualizations")

tab1, tab2, tab3 = st.tabs(["Utilization Analysis", "Quality Metrics", "Workload Profile"])

with tab1:
    # Nurse Utilization gauge
    fig_util = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = metrics_1['utilization'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Nurse Utilization"},
        delta = {'reference': 75, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 70], 'color': "lightgreen"},
                {'range': [70, 85], 'color': "yellow"},
                {'range': [85, 95], 'color': "orange"},
                {'range': [95, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    st.plotly_chart(fig_util, use_container_width=True)
    
    st.info("""
    **Target Range:** 70-85%
    - Below 70%: Underutilized (inefficient)
    - 70-85%: Optimal
    - 85-95%: High risk of errors
    - Above 95%: Critical - unsustainable
    """)

with tab2:
    # Quality metrics
    quality_metrics = pd.DataFrame({
        'Metric': ['Missed Care', 'Target'],
        'Value': [metrics_1['missed_care_pct'], 10]
    })
    
    fig_quality = px.bar(
        quality_metrics,
        x='Metric',
        y='Value',
        color='Metric',
        title="Missed Care vs. Target (10%)",
        color_discrete_map={'Missed Care': 'coral', 'Target': 'lightgreen'}
    )
    
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Wait time analysis
    nurse_range = range(1, 50)
    wait_times = [
        calculate_wait_time(n, arrival_rate, 0.5) 
        for n in nurse_range
    ]
    
    fig_wait = go.Figure()
    fig_wait.add_trace(go.Scatter(
        x=list(nurse_range),
        y=wait_times,
        mode='lines+markers',
        name='Wait Time',
        line=dict(color='purple', width=3)
    ))
    
    fig_wait.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target: 15 min")
    fig_wait.add_vline(x=num_nurses, line_dash="dot", line_color="blue", annotation_text="Current")
    
    fig_wait.update_layout(
        title="Wait Time vs. Number of Nurses",
        xaxis_title="Number of Nurses",
        yaxis_title="Average Wait Time (min)"
    )
    
    st.plotly_chart(fig_wait, use_container_width=True)

with tab3:
    # workload breakdown
    workload_data = pd.DataFrame({
        'Component': ['Patient Care', 'Documentation', 'Walking', 'Other'],
        'Hours': [
            shift_length * 0.5,
            shift_length * 0.25,
            shift_length * 0.15,
            shift_length * 0.1
        ]
    })
    
    fig_workload = px.pie(
        workload_data,
        values='Hours',
        names='Component',
        title=f"Estimated Time Allocation ({shift_length}h shift)"
    )
    
    st.plotly_chart(fig_workload, use_container_width=True)
    
    # Biomechanical load
    time_points = np.linspace(0, shift_length, 20)
    cumulative_load = [
        estimate_biomechanical_load(num_nurses, arrival_rate, t)
        for t in time_points
    ]
    
    fig_biomech = go.Figure()
    fig_biomech.add_trace(go.Scatter(
        x=time_points,
        y=cumulative_load,
        mode='lines',
        fill='tozeroy',
        name='Cumulative Load'
    ))
    
    fig_biomech.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Caution threshold")
    
    fig_biomech.update_layout(
        title="Cumulative Biomechanical Load Over Shift",
        xaxis_title="Hours into Shift",
        yaxis_title="Cumulative Spine Load (Nh)"
    )
    
    st.plotly_chart(fig_biomech, use_container_width=True)


# Recommendations based on defficiencies
st.header("Recommendations")

calculator = WorkloadCalculator()
recommendations = calculator.generate_recommendations(metrics_1, params_1)

for rec in recommendations:
    priority_emojis = {
        'High': '🔴',
        'Medium': '🟡',
        'Low': '🟢'
    }
    
    emoji = priority_emojis.get(rec['priority'], '⚪')
    
    st.markdown(f"""
    **{emoji} {rec['category']}** ({rec['priority']} Priority)  
    {rec['text']}
    """)

# Footer
st.markdown("---")
st.markdown("""
**Based on research by:**
- Neumann, W.P., et al. (2024). "Computer models in healthcare shed light on missed care"
- Qureshi, S.M., et al. (2019). "Predicting the effect of Nurse-Patient ratio on Nurse Workload"

**Developed by:** Yassine Adam

""")



