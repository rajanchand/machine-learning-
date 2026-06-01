"""
Page 7 — Explainability: SHAP feature importance analysis.
"""

import streamlit as st
import numpy as np
from streamlit_app.components.styles import section_header
from streamlit_app.components.charts import plot_feature_importance


def render():
    st.markdown("# 💡 Explainability")
    st.markdown("Understand which features drive model predictions using SHAP analysis.")
    st.markdown("---")

    results = st.session_state.get("results", {})
    data = st.session_state.get("data")

    if not results or not data:
        st.warning("⚠️ Train at least one model first.")
        return

    feature_names = data["feature_names"]
    X_test = data["X_test"]

    # Model selection — only IF and XGBoost have good SHAP support
    shap_models = [m for m in ["XGBoost", "Isolation Forest"] if m in st.session_state.get("trained_models", {})]

    if not shap_models:
        st.warning("⚠️ SHAP analysis requires a trained XGBoost or Isolation Forest model.")
        return

    selected = st.selectbox("Select model for SHAP analysis", shap_models)
    n_samples = st.slider("Number of samples to explain", 50, min(500, len(X_test)), 100)

    if st.button("🔬 Run SHAP Analysis", use_container_width=True):
        with st.spinner(f"Computing SHAP values for {selected} (this may take a moment)..."):
            try:
                X_sample = X_test[:n_samples]

                if selected == "XGBoost":
                    from src.explainability import explain_xgboost, get_top_features
                    model = st.session_state["trained_models"]["XGBoost"]
                    shap_values, explainer = explain_xgboost(model, X_sample, feature_names)
                    top = get_top_features(shap_values, feature_names)

                elif selected == "Isolation Forest":
                    from src.explainability import explain_isolation_forest, get_top_features
                    model = st.session_state["trained_models"]["Isolation Forest"]
                    shap_values, explainer = explain_isolation_forest(model, X_sample, feature_names)
                    top = get_top_features(shap_values, feature_names)

                # Store for later use
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_sample"] = X_sample
                st.session_state["shap_model"] = selected

                # ── Top Features Table ──
                section_header(f"Top {len(top)} Most Important Features")
                import pandas as pd
                top_df = pd.DataFrame(top, columns=["Feature", "Mean |SHAP|"])
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                # ── Feature Importance Bar Chart ──
                section_header("Feature Importance (SHAP)")
                importances = np.abs(shap_values).mean(axis=0)
                fig = plot_feature_importance(feature_names, importances, top_k=15)
                st.plotly_chart(fig, use_container_width=True)

                # ── SHAP Summary Plot (matplotlib) ──
                section_header("SHAP Summary Plot")
                import shap
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                fig = plt.gcf()
                fig.patch.set_facecolor("#ffffff")
                st.pyplot(fig)
                plt.close()

                st.success("✅ SHAP analysis complete!")

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ── XGBoost built-in feature importance (quick alternative) ──
    if "XGBoost" in st.session_state.get("trained_models", {}):
        section_header("XGBoost — Built-in Feature Importance")
        model = st.session_state["trained_models"]["XGBoost"]
        importances = model.feature_importances_
        fig = plot_feature_importance(feature_names, importances, top_k=20, title="XGBoost Feature Importance (Gain)")
        st.plotly_chart(fig, use_container_width=True)
