"""
Page 3 — Model Training: Select, configure, and train models with progress tracking.
"""

import streamlit as st
import numpy as np
from streamlit_app.components.styles import section_header
from streamlit_app.components.charts import plot_training_loss
from src.evaluation import compute_metrics


def render():
    st.markdown("# 🧠 Model Training")
    st.markdown("Select and train anomaly detection models.")
    st.markdown("---")

    if "data" not in st.session_state:
        st.warning("⚠️ No dataset loaded. Go to **📊 Data Explorer** first.")
        return

    data = st.session_state["data"]
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    n_features = data["n_features"]

    # Initialize results storage
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    # ── Model Selection ──
    section_header("Select Model")
    model_choice = st.selectbox(
        "Choose a model to train",
        ["Isolation Forest", "XGBoost", "LSTM Autoencoder", "Hybrid Ensemble"],
    )

    # ══════════════════════════════════════════════════════════════════
    # ISOLATION FOREST
    # ══════════════════════════════════════════════════════════════════
    if model_choice == "Isolation Forest":
        section_header("Isolation Forest — Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            contamination = st.slider("Contamination", 0.001, 0.1, 0.01, 0.001)
        with c2:
            n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)

        if st.button("🚀 Train Isolation Forest", use_container_width=True):
            with st.spinner("Training Isolation Forest..."):
                from src.models.isolation_forest import train_isolation_forest, predict_isolation_forest
                progress = st.progress(0, text="Fitting Isolation Forest...")

                model = train_isolation_forest(
                    X_train, contamination=contamination, n_estimators=n_estimators
                )
                progress.progress(60, text="Running predictions...")

                preds, scores = predict_isolation_forest(model, X_test)
                progress.progress(90, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, scores)
                progress.progress(100, text="Done!")

                st.session_state["trained_models"]["Isolation Forest"] = model
                st.session_state["results"]["Isolation Forest"] = {
                    "metrics": metrics, "preds": preds, "scores": scores,
                    "params": {"contamination": contamination, "n_estimators": n_estimators},
                }

                st.success(f"✅ Isolation Forest trained — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
                _show_quick_metrics(metrics)

    # ══════════════════════════════════════════════════════════════════
    # XGBOOST
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "XGBoost":
        section_header("XGBoost — Hyperparameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            xgb_n_est = st.slider("Number of estimators", 50, 1000, 300, 50)
        with c2:
            xgb_depth = st.slider("Max depth", 2, 15, 6)
        with c3:
            xgb_lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], 0.1)

        if st.button("🚀 Train XGBoost", use_container_width=True):
            with st.spinner("Training XGBoost..."):
                from src.models.xgboost_model import train_xgboost, predict_xgboost
                progress = st.progress(0, text="Fitting XGBoost classifier...")

                model = train_xgboost(
                    X_train, y_train, X_val, y_val,
                    n_estimators=xgb_n_est, max_depth=xgb_depth, learning_rate=xgb_lr,
                )
                progress.progress(70, text="Running predictions...")

                preds, proba = predict_xgboost(model, X_test)
                progress.progress(90, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, proba)
                progress.progress(100, text="Done!")

                st.session_state["trained_models"]["XGBoost"] = model
                st.session_state["results"]["XGBoost"] = {
                    "metrics": metrics, "preds": preds, "scores": proba,
                    "params": {"n_estimators": xgb_n_est, "max_depth": xgb_depth, "lr": xgb_lr},
                }

                st.success(f"✅ XGBoost trained — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
                _show_quick_metrics(metrics)

    # ══════════════════════════════════════════════════════════════════
    # LSTM AUTOENCODER
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "LSTM Autoencoder":
        section_header("LSTM Autoencoder — Hyperparameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            hidden_dim = st.selectbox("Hidden dim", [32, 64, 128], index=1)
            latent_dim = st.selectbox("Latent dim", [16, 32, 64], index=1)
        with c2:
            epochs = st.slider("Epochs", 10, 200, 50, 10)
            batch_size = st.selectbox("Batch size", [64, 128, 256, 512], index=2)
        with c3:
            use_attention = st.checkbox("Use Attention Mechanism", value=True)
            n_layers = st.selectbox("LSTM layers", [1, 2, 3], index=1)

        if st.button("🚀 Train LSTM Autoencoder", use_container_width=True):
            with st.spinner("Training LSTM Autoencoder (this may take a few minutes)..."):
                from src.models.lstm_autoencoder import train_lstm_autoencoder, predict_lstm_autoencoder

                # Train on NORMAL data only (semi-supervised)
                X_train_normal = X_train[y_train == 0]
                X_val_all = X_val  # validate on all data (normal + attack)

                progress_bar = st.progress(0, text="Training LSTM Autoencoder...")
                status_text = st.empty()

                def progress_callback(epoch, total, train_loss, val_loss):
                    pct = int(epoch / total * 80)
                    msg = f"Epoch {epoch}/{total} — Train Loss: {train_loss:.6f}"
                    if val_loss is not None:
                        msg += f" | Val Loss: {val_loss:.6f}"
                    progress_bar.progress(pct, text=msg)

                model, train_losses, val_losses, threshold = train_lstm_autoencoder(
                    X_train_normal, X_val_all,
                    n_features=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim,
                    n_layers=n_layers, use_attention=use_attention,
                    epochs=epochs, batch_size=batch_size,
                    progress_callback=progress_callback,
                )
                progress_bar.progress(85, text="Running predictions on test set...")

                preds, errors = predict_lstm_autoencoder(model, X_test, threshold)
                progress_bar.progress(95, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, errors)
                progress_bar.progress(100, text="Done!")

                st.session_state["trained_models"]["LSTM Autoencoder"] = model
                st.session_state["results"]["LSTM Autoencoder"] = {
                    "metrics": metrics, "preds": preds, "scores": errors,
                    "threshold": threshold,
                    "train_losses": train_losses, "val_losses": val_losses,
                    "params": {"hidden_dim": hidden_dim, "latent_dim": latent_dim,
                               "epochs": epochs, "use_attention": use_attention, "n_layers": n_layers},
                }

                st.success(f"✅ LSTM Autoencoder trained — F1: {metrics['F1-Score']:.4f} | Threshold: {threshold:.6f}")
                _show_quick_metrics(metrics)

                # Show training loss curve
                st.plotly_chart(plot_training_loss(train_losses, val_losses), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # HYBRID ENSEMBLE
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "Hybrid Ensemble":
        section_header("Hybrid Ensemble — Configuration")
        st.info("The ensemble requires all three base models to be trained first.")

        required = ["Isolation Forest", "XGBoost", "LSTM Autoencoder"]
        trained = [m for m in required if m in st.session_state.get("results", {})]
        missing = [m for m in required if m not in trained]

        if missing:
            st.warning(f"⚠️ Missing models: {', '.join(missing)}. Train them first.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            w_if = st.slider("IF weight", 0.0, 1.0, 0.25, 0.05)
        with c2:
            w_xgb = st.slider("XGBoost weight", 0.0, 1.0, 0.40, 0.05)
        with c3:
            w_ae = st.slider("LSTM-AE weight", 0.0, 1.0, 0.35, 0.05)

        total = w_if + w_xgb + w_ae
        if abs(total - 1.0) > 0.01:
            st.error(f"Weights must sum to 1.0 (current: {total:.2f})")
            return

        threshold = st.slider("Ensemble threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button("🚀 Run Hybrid Ensemble", use_container_width=True):
            from src.models.ensemble import ensemble_predict

            if_scores = st.session_state["results"]["Isolation Forest"]["scores"]
            xgb_scores = st.session_state["results"]["XGBoost"]["scores"]
            ae_scores = st.session_state["results"]["LSTM Autoencoder"]["scores"]

            preds, fused = ensemble_predict(
                if_scores, xgb_scores, ae_scores,
                weights=(w_if, w_xgb, w_ae), threshold=threshold,
            )
            metrics = compute_metrics(y_test, preds, fused)

            st.session_state["results"]["Hybrid Ensemble"] = {
                "metrics": metrics, "preds": preds, "scores": fused,
                "params": {"w_if": w_if, "w_xgb": w_xgb, "w_ae": w_ae, "threshold": threshold},
            }

            st.success(f"✅ Ensemble complete — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
            _show_quick_metrics(metrics)


def _show_quick_metrics(metrics: dict):
    """Display quick metric cards after training."""
    c1, c2, c3, c4, c5 = st.columns(5)
    from streamlit_app.components.styles import metric_card
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
