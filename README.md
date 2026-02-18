# MSIN0097 Predictive Analytics Coursework (Individual)

## Project structure
- `notebooks/`: analysis notebooks
- `src/`: helper functions (data prep, modeling, evaluation)
- `reports/`: figures + final report PDF
- `agent_logs/`: agent usage log + decision register (appendix material)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Data

Download the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file into:
data/raw/


