**Complete project documentation and goals/summary:**

# Healthcare Workload Simulator - Hybrid DES-ML framework for real-time nurse workload analysis and prediction.

## Streamlit Dashboard Link:

https://healthcare-workload-simulator-y-adam-neumann.streamlit.app/

## Overview

This project combines **Discrete Event Simulation** (rigorous modeling) with **Machine Learning** (fast predictions) to enable real-time exploration of healthcare staffing scenarios.

### Key Features

- **Instant Predictions** - ML meta-model predicts much faster than several DES iterations
- **Verification System** - Run full DES to validate ML predictions
- **Interactive Dashboard** - Web-based tool with real-time visualizations (graphs)
- **Smart Recommendations** - Evidence-based staffing and quality improvements
- **Comprehensive Metrics** - Utilization, missed care, wait times, biomechanical load

## Project Structure

```
healthcare-workload-simulator/
├── simulation/          # DES engine
├── ml_models/          # ML training pipeline
├── dashboard/          # Streamlit UI
├── scripts/            # Training/validation scripts
├── data/              # Generated training data
└── requirements.txt
```

## Research Foundation

Based on peer-reviewed research:

- **Qureshi, S.M., et al. (2019)** - DES methodology for nurse workload
- **Neumann, W.P., et al. (2024)** - Missed care as system design problem
- **Qureshi, S.M., et al. (2023)** - Biomechanics integration in simulation

## ML Model Performance

| Metric | R² Score | RMSE |
|--------|----------|------|
| Utilization | 0.92 | 3.2% |
| Missed Care | 0.91 | 2.8% |
| Wait Time | 0.89 | 2.1 min |

**Speed:** ML predictions much faster than full DE simulation


### Dashboard

1. Adjust parameters in sidebar (nurses, arrival rate, shift length, acuity)
2. View instant ML predictions
3. Click "Run DES" to verify predictions
4. Review recommendations


### Installation

# Clone repository
git clone https://github.com/adamyassine27/healthcare-workload-simulator.git

cd healthcare-workload-simulator

# Create virtual environment
python -m venv .venv

# Install dependencies
pip install -r requirements.txt


### Quick Start - Full ML Setup (Complete Pipeline) using Python Terminal
Note - the streamlit page is already trained on a 2,500 scenario size dataset. Skip to step #4 if that is sufficient. The streamlit demo video used a 100,000 scenario size dataset, didn't upload due to file sizes, uploaded the 2,500 size one. Steps 1-3 are to generate the dataset, train it, then validate it. Step 4 is to launch the dashboard locally.

# 1. Generate training data (currently set to 100,000 simulations - ~3hrs - lines 11+143)
cd scripts
python 01_generate_training_data.py

# 2. Train ML models
python 02_train_ml_models.py

# 3. Validate models
python 03_validate_models.py

# 4. Run dashboard with ML
cd ../dashboard

streamlit run app.py

## Author

**Yassine Adam**
- Email: yassine.adam.canada@gmail.com
- GitHub: @adamyassine27

## Acknowledgments

- Prof. W. Patrick Neumann (TMU HFE Lab) - Research methodology
- Qureshi et al. - DES implementation guidance
