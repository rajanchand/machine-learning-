"""
Page 3 — Model Training: Select, configure, and train models with progress tracking.
"""

import streamlit as st
import numpy as np
import pandas as pd
from streamlit_app.components.styles import section_header, metric_card
from streamlit_app.components.charts import plot_training_loss
from src.evaluation import compute_metrics


def render():
    st.markdown("# 🧠 Model Training & Feature Engineering")
    st.markdown("Configure feature selection, scale inputs, and train advanced anomaly detection architectures.")
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

    # ── Advanced Preprocessing Expandable Configuration ──
    section_header("⚙️ Feature Engineering & Preprocessing Pipeline")
    with st.expander("Advanced Configuration: Scaling, Feature Selection & PCA Subspace Projection", expanded=False):
        st.markdown("Customize your feature engineering pipeline below. Applying changes will re-scale, select, or project the current dataset.")
        
        use_robust = st.checkbox(
            "Use RobustScaler (Resilient to heavy outliers in network traffic)",
            value=st.session_state["data"].get("use_robust", True),
            help="RobustScaler uses median and IQR, making it extremely robust to network spikes and scan bursts."
        )
        
        col_mi, col_pca = st.columns(2)
        with col_mi:
            select_mi_features = st.checkbox(
                "Feature Selection via Mutual Information",
                value=st.session_state["data"].get("select_mi_features", False),
                help="Select the most informative features using non-linear Mutual Information scoring."
            )
            n_mi_features = st.slider(
                "Number of MI features to retain",
                min_value=5,
                max_value=max(5, len(st.session_state["data"]["df_features"].columns)),
                value=st.session_state["data"].get("n_mi_features", 15)
            )
        with col_pca:
            use_pca = st.checkbox(
                "PCA Dimensionality Reduction",
                value=st.session_state["data"].get("use_pca", False),
                help="Project high-dimensional features into a lower-dimensional principal component subspace."
            )
            n_pca_components = st.slider(
                "Number of PCA components",
                min_value=2,
                max_value=max(2, len(st.session_state["data"]["df_features"].columns) - 1),
                value=st.session_state["data"].get("n_pca_components", 10)
            )
            
        if st.button("🔄 Apply Custom Preprocessing Configurations", use_container_width=True):
            with st.spinner("Applying custom preprocessing pipeline..."):
                from src.preprocessing import encode_and_scale, split_data, save_preprocessing_artifacts
                
                df_features = st.session_state["data"]["df_features"]
                labels = st.session_state["data"]["labels"]
                test_size = st.session_state["data"]["test_size"]
                val_size = st.session_state["data"]["val_size"]
                
                X_scaled, scaler, label_encoders, feature_names = encode_and_scale(
                    df_features.copy(),
                    use_robust=use_robust,
                    select_mi_features=select_mi_features,
                    n_mi_features=n_mi_features,
                    y=labels.values,
                    use_pca=use_pca,
                    n_pca_components=n_pca_components
                )
                
                X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(
                    X_scaled, labels.values, test_size=test_size, val_size=val_size
                )
                
                # Save artifacts
                save_preprocessing_artifacts(scaler, label_encoders, feature_names)
                
                # Update session state
                st.session_state["data"].update({
                    "X_train": X_tr,
                    "X_val": X_va,
                    "X_test": X_te,
                    "y_train": y_tr,
                    "y_val": y_va,
                    "y_test": y_te,
                    "scaler": scaler,
                    "label_encoders": label_encoders,
                    "feature_names": feature_names,
                    "n_features": X_scaled.shape[1],
                    "use_robust": use_robust,
                    "select_mi_features": select_mi_features,
                    "n_mi_features": n_mi_features,
                    "use_pca": use_pca,
                    "n_pca_components": n_pca_components
                })
                st.success(f"🎉 Preprocessing applied successfully! Active features: {X_scaled.shape[1]}")
                st.rerun()

    st.markdown("---")

    # ── Model Selection ──
    section_header("Train Models")
    model_choice = st.selectbox(
        "Choose a model to train",
        [
            "Isolation Forest",
            "One-Class SVM",
            "XGBoost",
            "LSTM Autoencoder",
            "CNN-LSTM Autoencoder",
            "Hybrid Ensemble"
        ],
    )

    # ══════════════════════════════════════════════════════════════════
    # ISOLATION FOREST
    # ══════════════════════════════════════════════════════════════════
    if model_choice == "Isolation Forest":
        section_header("Isolation Forest — Hyperparameters (Unsupervised Baseline)")
        c1, c2 = st.columns(2)
        with c1:
            contamination = st.slider("Contamination (Target Outlier Ratio)", 0.001, 0.1, 0.01, 0.001)
        with c2:
            n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)

        if st.button("🚀 Train Isolation Forest", use_container_width=True):
            with st.spinner("Training Isolation Forest..."):
                from src.models.isolation_forest import train_and_save, predict_isolation_forest
                progress = st.progress(0, text="Fitting Isolation Forest...")

                model, metadata = train_and_save(
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

                st.success(f"✅ Isolation Forest trained & secured — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
                _show_quick_metrics(metrics)

    # ══════════════════════════════════════════════════════════════════
    # ONE-CLASS SVM
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "One-Class SVM":
        section_header("One-Class SVM — Hyperparameters (Unsupervised Baseline)")
        c1, c2 = st.columns(2)
        with c1:
            nu = st.slider("Nu (Contamination Boundary)", 0.001, 0.1, 0.05, 0.001)
            kernel = st.selectbox("Kernel type", ["rbf", "linear", "poly", "sigmoid"], index=0)
        with c2:
            gamma = st.selectbox("Gamma coefficient", ["scale", "auto"], index=0)

        if st.button("🚀 Train One-Class SVM", use_container_width=True):
            with st.spinner("Training One-Class SVM..."):
                from src.models.one_class_svm import train_and_save, predict_one_class_svm
                progress = st.progress(0, text="Filtering normal samples...")

                X_train_normal = X_train[y_train == 0]
                progress.progress(20, text="Fitting One-Class SVM (normal traffic only)...")

                model, metadata = train_and_save(
                    X_train_normal, nu=nu, kernel=kernel, gamma=gamma
                )
                progress.progress(70, text="Running predictions on test set...")

                preds, scores = predict_one_class_svm(model, X_test)
                progress.progress(90, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, scores)
                progress.progress(100, text="Done!")

                st.session_state["trained_models"]["One-Class SVM"] = model
                st.session_state["results"]["One-Class SVM"] = {
                    "metrics": metrics, "preds": preds, "scores": scores,
                    "params": {"nu": nu, "kernel": kernel, "gamma": gamma},
                }

                st.success(f"✅ One-Class SVM trained & secured — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
                _show_quick_metrics(metrics)

    # ══════════════════════════════════════════════════════════════════
    # XGBOOST
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "XGBoost":
        section_header("XGBoost — Hyperparameters (Supervised Baseline)")
        c1, c2, c3 = st.columns(3)
        with c1:
            xgb_n_est = st.slider("Number of estimators", 50, 1000, 300, 50)
        with c2:
            xgb_depth = st.slider("Max depth", 2, 15, 6)
        with c3:
            xgb_lr = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], 0.1)

        if st.button("🚀 Train XGBoost", use_container_width=True):
            with st.spinner("Training XGBoost..."):
                from src.models.xgboost_model import train_and_save, predict_xgboost
                progress = st.progress(0, text="Fitting XGBoost classifier...")

                model, metadata = train_and_save(
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

                st.success(f"✅ XGBoost trained & secured — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
                _show_quick_metrics(metrics)

    # ══════════════════════════════════════════════════════════════════
    # LSTM AUTOENCODER
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "LSTM Autoencoder":
        section_header("LSTM Autoencoder — Hyperparameters (Proposed Baseline)")
        c1, c2, c3 = st.columns(3)
        with c1:
            hidden_dim = st.selectbox("Hidden dimension", [32, 64, 128], index=1)
            latent_dim = st.selectbox("Latent dimension", [16, 32, 64], index=1)
        with c2:
            epochs = st.slider("Training Epochs", 10, 200, 50, 10)
            batch_size = st.selectbox("Minibatch size", [64, 128, 256, 512], index=2)
        with c3:
            use_attention = st.checkbox("Soft Attention Mechanism", value=True)
            n_layers = st.selectbox("LSTM hidden layers", [1, 2, 3], index=1)

        if st.button("🚀 Train LSTM Autoencoder", use_container_width=True):
            with st.spinner("Training LSTM Autoencoder (this may take a few minutes)..."):
                from src.models.lstm_autoencoder import train_and_save, predict_lstm_autoencoder

                X_train_normal = X_train[y_train == 0]
                X_val_all = X_val

                progress_bar = st.progress(0, text="Initializing LSTM Autoencoder...")

                def progress_callback(epoch, total, train_loss, val_loss):
                    pct = int(epoch / total * 80)
                    msg = f"Epoch {epoch}/{total} — Train Loss: {train_loss:.6f}"
                    if val_loss is not None:
                        msg += f" | Val Loss: {val_loss:.6f}"
                    progress_bar.progress(pct, text=msg)

                model, metadata = train_and_save(
                    X_train_normal, X_val_all,
                    n_features=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim,
                    n_layers=n_layers, use_attention=use_attention,
                    epochs=epochs, batch_size=batch_size,
                    progress_callback=progress_callback,
                )
                progress_bar.progress(85, text="Running reconstruction predictions on test set...")

                threshold = metadata["threshold"]
                preds, errors = predict_lstm_autoencoder(model, X_test, threshold)
                progress_bar.progress(95, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, errors)
                progress_bar.progress(100, text="Done!")

                st.session_state["trained_models"]["LSTM Autoencoder"] = model
                st.session_state["results"]["LSTM Autoencoder"] = {
                    "metrics": metrics, "preds": preds, "scores": errors,
                    "threshold": threshold,
                    "train_losses": metadata["train_losses"], "val_losses": metadata["val_losses"],
                    "params": {"hidden_dim": hidden_dim, "latent_dim": latent_dim,
                               "epochs": epochs, "use_attention": use_attention, "n_layers": n_layers},
                }

                st.success(f"✅ LSTM Autoencoder trained & secured — F1: {metrics['F1-Score']:.4f} | Dynamic Threshold: {threshold:.6f}")
                _show_quick_metrics(metrics)

                # Show training loss curve
                st.plotly_chart(plot_training_loss(metadata["train_losses"], metadata["val_losses"]), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # CNN-LSTM AUTOENCODER
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "CNN-LSTM Autoencoder":
        section_header("CNN-LSTM Autoencoder — Hyperparameters (Proposed Hybrid Model)")
        st.markdown(
            "This model applies **1D spatial convolutions** across tabular features to extract structural groups, "
            "followed by **LSTM sequence modeling** and a **Soft sequential Attention mechanism**."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            conv_channels = st.selectbox("CNN Conv Channels", [8, 16, 32], index=1)
            hidden_dim = st.selectbox("LSTM Hidden dimension", [32, 64, 128], index=1)
        with c2:
            latent_dim = st.selectbox("Latent bottleneck size", [16, 32, 64], index=1)
            epochs = st.slider("Training Epochs", 10, 200, 50, 10)
        with c3:
            use_attention = st.checkbox("Soft Attention Mechanism", value=True)
            n_layers = st.selectbox("LSTM hidden layers", [1, 2, 3], index=1)

        if st.button("🚀 Train CNN-LSTM Autoencoder", use_container_width=True):
            with st.spinner("Training proposed CNN-LSTM Autoencoder..."):
                from src.models.cnn_lstm_autoencoder import train_and_save, predict_cnn_lstm_autoencoder

                X_train_normal = X_train[y_train == 0]
                X_val_all = X_val

                progress_bar = st.progress(0, text="Initializing Proposed Architecture...")

                def progress_callback(epoch, total, train_loss, val_loss):
                    pct = int(epoch / total * 80)
                    msg = f"Epoch {epoch}/{total} — Train Loss: {train_loss:.6f}"
                    if val_loss is not None:
                        msg += f" | Val Loss: {val_loss:.6f}"
                    progress_bar.progress(pct, text=msg)

                model, metadata = train_and_save(
                    X_train_normal, X_val_all,
                    n_features=n_features, conv_channels=conv_channels,
                    hidden_dim=hidden_dim, latent_dim=latent_dim,
                    n_layers=n_layers, use_attention=use_attention,
                    epochs=epochs, batch_size=256,
                    progress_callback=progress_callback,
                )
                progress_bar.progress(85, text="Computing spatial-temporal reconstructions...")

                threshold = metadata["threshold"]
                preds, errors = predict_cnn_lstm_autoencoder(model, X_test, threshold)
                progress_bar.progress(95, text="Computing metrics...")

                metrics = compute_metrics(y_test, preds, errors)
                progress_bar.progress(100, text="Done!")

                st.session_state["trained_models"]["CNN-LSTM Autoencoder"] = model
                st.session_state["results"]["CNN-LSTM Autoencoder"] = {
                    "metrics": metrics, "preds": preds, "scores": errors,
                    "threshold": threshold,
                    "train_losses": metadata["train_losses"], "val_losses": metadata["val_losses"],
                    "params": {"conv_channels": conv_channels, "hidden_dim": hidden_dim,
                               "latent_dim": latent_dim, "epochs": epochs,
                               "use_attention": use_attention, "n_layers": n_layers},
                }

                st.success(f"✅ Proposed CNN-LSTM Autoencoder trained & secured — F1: {metrics['F1-Score']:.4f} | Dynamic Threshold: {threshold:.6f}")
                _show_quick_metrics(metrics)

                # Show training loss curve
                st.plotly_chart(plot_training_loss(metadata["train_losses"], metadata["val_losses"]), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # HYBRID ENSEMBLE
    # ══════════════════════════════════════════════════════════════════
    elif model_choice == "Hybrid Ensemble":
        section_header("Hybrid Ensemble — Adaptive Score Fusion Panel")
        st.markdown(
            "Fuse scores from one unsupervised baseline, one supervised baseline, and one deep autoencoder model "
            "using **weighted min-max normalization average fusion** and an **adaptive fusion threshold**."
        )

        st.info("The hybrid ensemble requires the component models to be trained first.")

        # Let the user configure which component models to fuse
        st.markdown("### 1. Select Ensemble Base Components")
        c_models = st.columns(3)
        with c_models[0]:
            unsup_choice = st.selectbox("Unsupervised Baseline", ["Isolation Forest", "One-Class SVM"])
        with c_models[1]:
            sup_choice = st.selectbox("Supervised Baseline", ["XGBoost"])
        with c_models[2]:
            ae_choice = st.selectbox("Deep Autoencoder", ["LSTM Autoencoder", "CNN-LSTM Autoencoder"])

        required = [unsup_choice, sup_choice, ae_choice]
        trained = [m for m in required if m in st.session_state.get("results", {})]
        missing = [m for m in required if m not in trained]

        if missing:
            st.warning(f"⚠️ Missing component models: **{', '.join(missing)}**. Please select and train them in this session first.")
            return

        st.markdown("### 2. Configure Weights & Adaptive Threshold")
        c1, c2, c3 = st.columns(3)
        with c1:
            w_unsup = st.slider(f"{unsup_choice} weight", 0.0, 1.0, 0.25, 0.05)
        with c2:
            w_sup = st.slider(f"{sup_choice} weight", 0.0, 1.0, 0.40, 0.05)
        with c3:
            w_ae = st.slider(f"{ae_choice} weight", 0.0, 1.0, 0.35, 0.05)

        total = w_unsup + w_sup + w_ae
        if abs(total - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to exactly 1.0 (current sum: {total:.2f})")
            return

        thresh_method = st.selectbox(
            "Adaptive Threshold Estimation Method",
            ["fusion", "gaussian", "percentile"],
            index=0,
            help="Choose the mathematical estimator for the fused anomaly boundary: "
                 "Gaussian standard deviation spikes, extreme percentiles (99.5th), or an average of both (fusion)."
        )

        if st.button("🚀 Execute Hybrid Ensemble Fusion", use_container_width=True):
            from src.models.ensemble import ensemble_predict

            unsup_scores = st.session_state["results"][unsup_choice]["scores"]
            sup_scores = st.session_state["results"][sup_choice]["scores"]
            ae_scores = st.session_state["results"][ae_choice]["scores"]

            # Run prediction fusion
            preds, fused = ensemble_predict(
                unsup_scores, sup_scores, ae_scores,
                weights=(w_unsup, w_sup, w_ae),
                threshold=None, # auto-compute
                threshold_method=thresh_method,
            )
            metrics = compute_metrics(y_test, preds, fused)

            st.session_state["results"]["Hybrid Ensemble"] = {
                "metrics": metrics, "preds": preds, "scores": fused,
                "params": {
                    "unsup_model": unsup_choice, "w_unsup": w_unsup,
                    "sup_model": sup_choice, "w_sup": w_sup,
                    "ae_model": ae_choice, "w_ae": w_ae,
                    "threshold_method": thresh_method
                },
            }

            st.success(f"✅ Ensemble Fusion completed — F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
            _show_quick_metrics(metrics)


def _show_quick_metrics(metrics: dict):
    """Display quick metric cards after training."""
    c1, c2, c3, c4, c5 = st.columns(5)
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
