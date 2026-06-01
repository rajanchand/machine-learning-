"""
app.py — Main Streamlit entry point for NetShield.

Run with:  streamlit run app.py
"""

import streamlit as st
from streamlit_app.auth import login_form, logout
from streamlit_app.components.styles import inject_custom_css

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetShield — ML Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
inject_custom_css()

# ── Auth Gate ────────────────────────────────────────────────────────
if not login_form():
    st.stop()

# Stateful Session Validation
from streamlit_app.auth import check_session_timeout
if not check_session_timeout():
    st.stop()

# ── Navigation ───────────────────────────────────────────────────────
PAGES = {
    "🏠 Dashboard":         "streamlit_app.pages.1_dashboard",
    "📊 Data Explorer":     "streamlit_app.pages.2_data_explorer",
    "🧠 Model Training":    "streamlit_app.pages.3_training",
    "🔍 Prediction":        "streamlit_app.pages.4_prediction",
    "📈 Results":           "streamlit_app.pages.5_results",
    "⚖️ Model Comparison":  "streamlit_app.pages.6_comparison",
    "💡 Explainability":    "streamlit_app.pages.7_explainability",
    "📄 Report":            "streamlit_app.pages.8_report",
}

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛡️ NetShield")
    st.caption(f"Signed in as **{st.session_state.get('display_name', 'User')}**")
    st.markdown("---")

    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("---")
    # System info
    with st.expander("💾 System Info"):
        from src.utils import list_saved_models
        models = list_saved_models()
        st.write(f"**Saved models:** {len(models)}")
        for m in models:
            st.caption(f"• {m['name']} ({m['size_mb']} MB)")

        if "data" in st.session_state:
            info = st.session_state["data"]
            st.write(f"**Dataset:** {info.get('dataset_format', 'N/A')}")
            st.write(f"**Features:** {info.get('n_features', 'N/A')}")

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        logout()

# ── Page Router ──────────────────────────────────────────────────────
import importlib

module_path = PAGES[selected]
page_module = importlib.import_module(module_path)
page_module.render()
