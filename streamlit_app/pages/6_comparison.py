"""
Page 6 — Model Comparison: Side-by-side metrics, radar chart, bar chart.
"""

import streamlit as st
import pandas as pd
from streamlit_app.components.styles import section_header
from streamlit_app.components.charts import plot_model_comparison_radar
from src.evaluation import compare_models, ablation_study
import plotly.graph_objects as go


def render():
    st.markdown("# ⚖️ Model Comparison")
    st.markdown("Compare all trained models side-by-side.")
    st.markdown("---")

    results = st.session_state.get("results", {})
    if len(results) < 2:
        st.warning("⚠️ Train at least 2 models for comparison. Go to **🧠 Model Training**.")
        return

    # ── Metrics Table ──
    section_header("Performance Comparison Table")
    metrics_dict = {name: r["metrics"] for name, r in results.items()}
    comparison_df = compare_models(metrics_dict)

    # Highlight best values
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="#2d1b69",
        ).highlight_min(
            subset=["FPR", "FNR"],
            color="#1b3d2d",
        ).format(precision=4),
        use_container_width=True,
    )

    # ── Radar Chart ──
    section_header("Radar Chart Comparison")
    fig = plot_model_comparison_radar(metrics_dict)
    st.plotly_chart(fig, use_container_width=True)

    # ── Bar Chart ──
    section_header("Metric Bar Chart")
    metric_choice = st.selectbox("Select metric", ["F1-Score", "Accuracy", "Precision", "Recall", "ROC-AUC", "FPR"])

    colors = ["#7c3aed", "#34d399", "#fbbf24", "#60a5fa"]
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics_dict.keys()),
            y=[m.get(metric_choice, 0) or 0 for m in metrics_dict.values()],
            marker_color=colors[:len(metrics_dict)],
            text=[f"{m.get(metric_choice, 0):.4f}" for m in metrics_dict.values()],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=f"{metric_choice} Comparison",
        template="plotly_dark", height=400,
        yaxis_title=metric_choice,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Ablation Study ──
    if "LSTM Autoencoder" in results and len(results) >= 3:
        section_header("Ablation Study")
        st.markdown("Comparing the proposed model (LSTM Autoencoder) against baselines and the ensemble.")

        base = results["LSTM Autoencoder"]["metrics"]
        variants = {name: r["metrics"] for name, r in results.items() if name != "LSTM Autoencoder"}
        ablation_df = ablation_study(base, variants)

        # Show only key columns
        display_cols = [c for c in ablation_df.columns if not c.endswith("_Δ")]
        delta_cols = [c for c in ablation_df.columns if c.endswith("_Δ")]

        tab1, tab2 = st.tabs(["Absolute Values", "Delta (Δ) vs Proposed"])
        with tab1:
            st.dataframe(ablation_df[display_cols].style.format(precision=4), use_container_width=True)
        with tab2:
            if delta_cols:
                st.dataframe(ablation_df[delta_cols].style.format(precision=4), use_container_width=True)
