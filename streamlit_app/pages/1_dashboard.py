"""
Page 1 — Dashboard: Project overview and system status.
"""

import streamlit as st
from streamlit_app.components.styles import metric_card, section_header, badge
from src.utils import list_saved_models


def render():
    st.markdown("# 🏠 Dashboard")
    st.markdown("**ML-Based Network Anomaly Detection** — Master's Dissertation Project")
    st.markdown("---")

    # ── Quick Stats ──
    section_header("System Overview")
    c1, c2, c3, c4 = st.columns(4)

    models = list_saved_models()
    data_loaded = "data" in st.session_state
    results_exist = "results" in st.session_state

    with c1:
        st.markdown(metric_card("Saved Models", len(models)), unsafe_allow_html=True)
    with c2:
        status = badge("Loaded", "success") if data_loaded else badge("Not Loaded", "warning")
        st.markdown(metric_card("Dataset", status), unsafe_allow_html=True)
    with c3:
        trained = sum(1 for r in st.session_state.get("results", {}).values() if r)
        st.markdown(metric_card("Trained Models", trained), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Status", badge("Ready", "success")), unsafe_allow_html=True)

    st.markdown("")

    # ── Project Overview ──
    section_header("About This Project")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        This system implements a **complete ML pipeline** for detecting network intrusions
        and anomalies in network traffic data. It supports:

        - **CIC-IDS2017** and **UNSW-NB15** benchmark datasets
        - Three detection models: **Isolation Forest**, **XGBoost**, and **LSTM Autoencoder**
        - A **hybrid ensemble** that fuses all three models (proposed novelty)
        - **SHAP-based explainability** for feature importance analysis
        - **Downloadable evaluation reports** (HTML format)

        #### Workflow
        1. **Upload & Explore** your dataset in the Data Explorer
        2. **Train** one or more models with configurable hyperparameters
        3. **Evaluate** performance with comprehensive metrics and visualisations
        4. **Compare** models side-by-side with radar charts
        5. **Explain** predictions using SHAP feature attribution
        6. **Generate** a full dissertation-quality report
        """)

    with col2:
        st.markdown("""
        #### Tech Stack
        - 🐍 Python 3.9+
        - 🔥 PyTorch (LSTM-AE)
        - 🌲 XGBoost
        - 🧪 scikit-learn
        - 📊 Plotly / Seaborn
        - 💡 SHAP
        - 🎨 Streamlit

        #### Models
        | Model | Type |
        |-------|------|
        | Isolation Forest | Unsupervised |
        | XGBoost | Supervised |
        | LSTM Autoencoder | Semi-supervised |
        | Hybrid Ensemble | Fusion |
        """)

    # ── Quick Start Guide ──
    section_header("Quick Start Guide")
    with st.expander("📖 How to use this application", expanded=False):
        st.markdown("""
        1. Navigate to **📊 Data Explorer** and upload a CSV dataset (CIC-IDS2017 or UNSW-NB15)
        2. The system will automatically detect the format, clean the data, and split it
        3. Go to **🧠 Model Training** to train one or more models
        4. View results in **📈 Results** — confusion matrices, ROC curves, score distributions
        5. Compare all trained models in **⚖️ Model Comparison**
        6. Explore feature importance in **💡 Explainability** (SHAP analysis)
        7. Generate a downloadable report in **📄 Report**

        **Default credentials:** `admin` / `admin`
        """)
