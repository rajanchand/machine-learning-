"""
Page 5 — Results: Detailed visualisation of model evaluation metrics.
"""

import streamlit as st
import numpy as np
from streamlit_app.components.styles import section_header, metric_card
from streamlit_app.components.charts import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_anomaly_score_distribution,
)
from src.evaluation import get_classification_report


def render():
    st.markdown("# 📈 Results")
    st.markdown("Detailed evaluation of trained models.")
    st.markdown("---")

    if not st.session_state.get("results"):
        st.warning("⚠️ No models trained yet. Go to **🧠 Model Training**.")
        return

    data = st.session_state["data"]
    y_test = data["y_test"]

    model_names = list(st.session_state["results"].keys())
    selected_model = st.selectbox("Select model to view", model_names)

    result = st.session_state["results"][selected_model]
    metrics = result["metrics"]
    preds = result["preds"]
    scores = result["scores"]

    # ── Metric Cards ──
    section_header(f"{selected_model} — Performance Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card("Accuracy", f"{metrics['Accuracy']:.4f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Precision", f"{metrics['Precision']:.4f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Recall", f"{metrics['Recall']:.4f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("F1-Score", f"{metrics['F1-Score']:.4f}"), unsafe_allow_html=True)
    with c5:
        auc = metrics.get("ROC-AUC")
        st.markdown(metric_card("ROC-AUC", f"{auc:.4f}" if auc else "N/A"), unsafe_allow_html=True)
    with c6:
        st.markdown(metric_card("FPR", f"{metrics['FPR']:.4f}"), unsafe_allow_html=True)

    st.markdown("")

    # ── Visualisations ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Confusion Matrix", "ROC Curve", "PR Curve", "Score Distribution", "Classification Report"
    ])

    with tab1:
        fig = plot_confusion_matrix(y_test, preds, title=f"{selected_model} — Confusion Matrix")
        st.pyplot(fig)

    with tab2:
        if scores is not None:
            fig = plot_roc_curve(y_test, scores, selected_model)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve requires continuous scores.")

    with tab3:
        if scores is not None:
            fig = plot_pr_curve(y_test, scores, selected_model)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        threshold = result.get("threshold")
        fig = plot_anomaly_score_distribution(scores, y_test, threshold, f"{selected_model} — Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        report = get_classification_report(y_test, preds)
        st.code(report, language="text")

    # ── Top Anomalous Flows ──
    section_header("Top 20 Most Anomalous Flows")
    import pandas as pd
    top_indices = np.argsort(scores)[::-1][:20]
    top_df = pd.DataFrame({
        "Rank": range(1, 21),
        "Index": top_indices,
        "Anomaly Score": [f"{scores[i]:.6f}" for i in top_indices],
        "True Label": ["Attack" if y_test[i] == 1 else "Normal" for i in top_indices],
        "Prediction": ["Attack" if preds[i] == 1 else "Normal" for i in top_indices],
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)
