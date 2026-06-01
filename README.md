# 🛡️ NetShield — ML Network Anomaly Detection

A professional, dissertation-quality network anomaly detection system with a Streamlit web interface. Implements multiple ML models for detecting network intrusions in CIC-IDS2017, UNSW-NB15, and custom datasets.

## Features

- **Multiple Models**: Isolation Forest, XGBoost, LSTM Autoencoder (PyTorch), Hybrid Ensemble
- **Attention Mechanism**: Optional self-attention in the LSTM Autoencoder (novelty)
- **8-Page Streamlit UI**: Dashboard, Data Explorer, Training, Prediction, Results, Comparison, Explainability, Report
- **SHAP Explainability**: Feature importance analysis with TreeExplainer and KernelExplainer
- **Session-based Login**: Lightweight bcrypt authentication
- **Downloadable Reports**: HTML evaluation reports with embedded charts
- **Ablation Study**: Compare model variants for dissertation evaluation

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Default credentials: `admin` / `admin`

## Project Structure

```
├── app.py                      # Streamlit entry point
├── config.yaml                 # Auth & app configuration
├── requirements.txt
├── streamlit_app/              # UI layer
│   ├── auth.py                 # Login/logout
│   ├── pages/                  # 8 application pages
│   │   ├── 1_dashboard.py
│   │   ├── 2_data_explorer.py
│   │   ├── 3_training.py
│   │   ├── 4_prediction.py
│   │   ├── 5_results.py
│   │   ├── 6_comparison.py
│   │   ├── 7_explainability.py
│   │   └── 8_report.py
│   └── components/             # Reusable UI components
│       ├── styles.py           # Custom CSS theming
│       └── charts.py           # Plotly/Matplotlib charts
├── src/                        # ML pipeline
│   ├── preprocessing.py        # Data loading, cleaning, scaling
│   ├── feature_engineering.py  # Feature selection, PCA
│   ├── evaluation.py           # Metrics, confusion matrix, ROC
│   ├── explainability.py       # SHAP wrappers
│   ├── report_generator.py     # HTML report generation
│   ├── utils.py                # Logging, model I/O, seeding
│   └── models/
│       ├── isolation_forest.py # Unsupervised baseline
│       ├── xgboost_model.py    # Supervised baseline
│       ├── lstm_autoencoder.py # Proposed model (PyTorch)
│       └── ensemble.py         # Hybrid fusion (novelty)
├── models/saved/               # Saved model artifacts
├── data/                       # Dataset storage
└── src/realtime/               # Archived real-time FastAPI service
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| Isolation Forest | Unsupervised | Anomaly detection via random partitioning |
| XGBoost | Supervised | Gradient-boosted classifier with class balancing |
| LSTM Autoencoder | Semi-supervised | Trained on normal traffic; high reconstruction error = anomaly |
| Hybrid Ensemble | Fusion | Weighted score fusion of all three models |

## Supported Datasets

- **CIC-IDS2017** — Auto-detected by column names (`Flow Duration`, etc.)
- **UNSW-NB15** — Auto-detected by column names (`sbytes`, `dbytes`, etc.)
- **Custom CSV** — Any CSV with a `Label` or `label` column
- **PCAP files** — Basic packet-level feature extraction via Scapy
