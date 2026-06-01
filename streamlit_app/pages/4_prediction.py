"""
Page 4 — Prediction: Batch or single-flow anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_app.components.styles import section_header, badge


def render():
    st.markdown("# 🔍 Prediction")
    st.markdown("Run anomaly detection on new network traffic data.")
    st.markdown("---")

    if "data" not in st.session_state or not st.session_state.get("results"):
        st.warning("⚠️ Train at least one model first via **🧠 Model Training**.")
        return

    data = st.session_state["data"]
    available_models = list(st.session_state["results"].keys())

    model_name = st.selectbox("Select model for prediction", available_models)

    tab1, tab2 = st.tabs(["📁 Batch Prediction (CSV)", "✏️ Single Flow"])

    # ── Batch Prediction ──
    with tab1:
        section_header("Upload CSV for Batch Prediction")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="pred_upload")

        if uploaded:
            try:
                from src.preprocessing import load_csv, clean_data, encode_and_scale
                from src.utils import load_preprocessing_artifacts
                import tempfile, os

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name

                df = load_csv(tmp_path)
                os.unlink(tmp_path)

                st.dataframe(df.head(20), use_container_width=True)

                if st.button("🔍 Run Predictions", key="batch_pred"):
                    with st.spinner("Running predictions..."):
                        # Use stored scaler
                        artifacts = load_preprocessing_artifacts()
                        scaler = artifacts["scaler"]
                        feature_names = artifacts["feature_names"]

                        # Prepare features (drop label if present, drop IPs)
                        for col in ["Label", "label", "class", "target", "attack_cat"]:
                            if col in df.columns:
                                df = df.drop(columns=[col])
                        ip_cols = [c for c in df.columns if "ip" in c.lower() or "addr" in c.lower()]
                        df = df.drop(columns=ip_cols, errors="ignore")

                        # Encode categoricals
                        for col, le in artifacts.get("label_encoders", {}).items():
                            if col in df.columns:
                                df[col] = df[col].astype(str).apply(
                                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                                )

                        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

                        # Align columns
                        for c in feature_names:
                            if c not in df.columns:
                                df[c] = 0
                        df = df[feature_names]

                        X_pred = scaler.transform(df.values)

                        # Run model
                        result = st.session_state["results"][model_name]
                        scores = _run_prediction(model_name, X_pred)

                        if scores is not None:
                            # Display results
                            results_df = df.copy()
                            results_df["Anomaly_Score"] = scores
                            threshold = result.get("threshold", np.percentile(scores, 95))
                            results_df["Prediction"] = ["🔴 Attack" if s > threshold else "🟢 Normal" for s in scores]

                            st.dataframe(
                                results_df.sort_values("Anomaly_Score", ascending=False).head(50),
                                use_container_width=True,
                            )

                            n_anomaly = (scores > threshold).sum()
                            st.info(f"Detected **{n_anomaly}** anomalies out of {len(scores)} flows (threshold: {threshold:.4f})")

            except Exception as e:
                st.error(f"Error: {e}")

    # ── Single Flow ──
    with tab2:
        section_header("Enter Flow Features Manually")
        feature_names = data["feature_names"]

        if len(feature_names) > 20:
            st.info(f"Showing first 20 of {len(feature_names)} features. Use batch mode for full feature sets.")
            show_features = feature_names[:20]
        else:
            show_features = feature_names

        cols = st.columns(4)
        values = {}
        for i, fname in enumerate(show_features):
            with cols[i % 4]:
                values[fname] = st.number_input(fname, value=0.0, key=f"feat_{fname}")

        if st.button("🔍 Predict Single Flow", key="single_pred"):
            # Build feature vector
            x = np.zeros(len(feature_names))
            for i, fname in enumerate(feature_names):
                x[i] = values.get(fname, 0.0)

            x_scaled = data["scaler"].transform(x.reshape(1, -1))
            scores = _run_prediction(model_name, x_scaled)

            if scores is not None:
                score = scores[0]
                result = st.session_state["results"][model_name]
                threshold = result.get("threshold", 0.5)
                is_anomaly = score > threshold

                if is_anomaly:
                    st.error(f"🔴 **ANOMALY DETECTED** — Score: {score:.6f} (Threshold: {threshold:.4f})")
                else:
                    st.success(f"🟢 **Normal Traffic** — Score: {score:.6f} (Threshold: {threshold:.4f})")


def _run_prediction(model_name: str, X: np.ndarray):
    """Run prediction using the appropriate model."""
    try:
        if model_name == "Isolation Forest":
            model = st.session_state["trained_models"]["Isolation Forest"]
            from src.models.isolation_forest import predict_isolation_forest
            _, scores = predict_isolation_forest(model, X)
            return scores

        elif model_name == "XGBoost":
            model = st.session_state["trained_models"]["XGBoost"]
            from src.models.xgboost_model import predict_xgboost
            _, scores = predict_xgboost(model, X)
            return scores

        elif model_name == "LSTM Autoencoder":
            model = st.session_state["trained_models"]["LSTM Autoencoder"]
            from src.models.lstm_autoencoder import compute_reconstruction_errors
            scores = compute_reconstruction_errors(model, X)
            return scores

        elif model_name == "Hybrid Ensemble":
            required = ["Isolation Forest", "XGBoost", "LSTM Autoencoder"]
            missing = [m for m in required if m not in st.session_state.get("trained_models", {})]
            if missing:
                st.error(f"Ensemble prediction requires all three base models to be trained. Missing: {', '.join(missing)}")
                return None
            
            # Predict each model individually
            model_if = st.session_state["trained_models"]["Isolation Forest"]
            model_xgb = st.session_state["trained_models"]["XGBoost"]
            model_ae = st.session_state["trained_models"]["LSTM Autoencoder"]
            
            from src.models.isolation_forest import predict_isolation_forest
            from src.models.xgboost_model import predict_xgboost
            from src.models.lstm_autoencoder import compute_reconstruction_errors
            from src.models.ensemble import ensemble_predict
            
            _, if_scores = predict_isolation_forest(model_if, X)
            _, xgb_scores = predict_xgboost(model_xgb, X)
            ae_scores = compute_reconstruction_errors(model_ae, X)
            
            # Use configured weights
            ensemble_result = st.session_state["results"]["Hybrid Ensemble"]
            params = ensemble_result["params"]
            weights = (params["w_if"], params["w_xgb"], params["w_ae"])
            threshold = params["threshold"]
            
            _, fused = ensemble_predict(if_scores, xgb_scores, ae_scores, weights=weights, threshold=threshold)
            return fused

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
