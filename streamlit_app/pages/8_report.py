"""
Page 8 — Report: Generate and download a full HTML evaluation report complete with digital seals.
"""

import streamlit as st
import os
import matplotlib.pyplot as plt
from streamlit_app.components.styles import section_header
from streamlit_app.components.charts import plot_confusion_matrix
from src.report_generator import generate_html_report, fig_to_base64
from src.evaluation import compare_models, ablation_study


def get_model_signature(model_name: str) -> str:
    """Read the trained model's cryptographic sidecar SHA-256 hash."""
    mapping = {
        "Isolation Forest": "isolation_forest.joblib",
        "One-Class SVM": "one_class_svm.joblib",
        "XGBoost": "xgboost.joblib",
        "LSTM Autoencoder": "lstm_autoencoder.pt",
        "CNN-LSTM Autoencoder": "cnn_lstm_autoencoder.pt",
    }
    filename = mapping.get(model_name)
    if not filename:
        return "N/A (Virtual Dynamic Ensemble)"
    
    from src.utils import MODELS_DIR
    sig_path = os.path.join(MODELS_DIR, filename + ".sha256")
    if os.path.exists(sig_path):
        try:
            with open(sig_path, "r") as f:
                return f.read().strip()
        except Exception:
            return "Error reading signature sidecar"
    return "No signature file found on disk"


def render():
    st.markdown("# 📄 Report Generator")
    st.markdown("Generate a comprehensive, cryptographically signed evaluation report for your dissertation.")
    st.markdown("---")

    results = st.session_state.get("results", {})
    data = st.session_state.get("data")

    if not results or not data:
        st.warning("⚠️ No evaluation results available. Train models and view results first via **🧠 Model Training**.")
        return

    section_header("Report Configuration")

    title = st.text_input("Report Dissertation Title", "Advanced Machine Learning-based Network Anomaly Detection System — Evaluation Report")

    models_to_include = st.multiselect(
        "Select models to incorporate in report", list(results.keys()), default=list(results.keys())
    )

    include_figures = st.checkbox("Include performance visualisations (Confusion Matrix & ROC curves)", value=True)

    if st.button("📄 Compile Interactive Report", use_container_width=True):
        with st.spinner("Compiling cryptographic report and rendering charts..."):
            y_test = data["y_test"]

            # Dataset info
            dataset_info = {
                "Dataset Format": data["dataset_format"],
                "Preprocessed Features": data["n_features"],
                "Total Record Samples": sum(data["class_distribution"].values()),
                "Normal Samples": data["class_distribution"]["normal"],
                "Attack Samples": data["class_distribution"]["attack"],
                "Train Split Size": len(data["y_train"]),
                "Validation Split Size": len(data["y_val"]),
                "Test Split Size": len(data["y_test"]),
            }

            # Model results
            model_results = {}
            cryptographic_signatures = {}
            for name in models_to_include:
                if name in results:
                    model_results[name] = results[name]["metrics"]
                    cryptographic_signatures[name] = get_model_signature(name)

            # Comparison table HTML
            comparison_html = ""
            if len(model_results) >= 2:
                df = compare_models(model_results)
                comparison_html = df.to_html(classes="comparison-table")

            # Ablation table HTML
            ablation_html = ""
            ablation_base = None
            for candidate in ["CNN-LSTM Autoencoder", "LSTM Autoencoder"]:
                if candidate in model_results:
                    ablation_base = candidate
                    break
            
            if ablation_base and len(model_results) >= 2:
                base_metrics = model_results[ablation_base]
                variants = {name: m for name, m in model_results.items() if name != ablation_base}
                ablation_df = ablation_study(base_metrics, variants)
                ablation_html = ablation_df.to_html(classes="ablation-table")

            # Figures
            figures = {}
            if include_figures:
                for name in models_to_include:
                    if name in results:
                        r = results[name]
                        try:
                            # 1. Confusion matrix (Styled for Light Theme)
                            fig = plot_confusion_matrix(y_test, r["preds"], f"{name} — Confusion Matrix")
                            figures[f"{name} — Confusion Matrix"] = fig_to_base64(fig)
                            plt.close(fig)

                            # 2. ROC curve (Styled for Light Theme matplotlib)
                            if r["scores"] is not None:
                                from sklearn.metrics import roc_curve, auc
                                fpr, tpr, _ = roc_curve(y_test, r["scores"])
                                roc_auc = auc(fpr, tpr)
                                
                                fig2, ax = plt.subplots(figsize=(7, 5))
                                ax.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"AUC={roc_auc:.4f}")
                                ax.plot([0, 1], [0, 1], "--", color="#64748b")
                                ax.set_xlabel("False Positive Rate (FPR)")
                                ax.set_ylabel("True Positive Rate (TPR)")
                                ax.set_title(f"{name} — ROC Curve")
                                ax.legend(loc="lower right")
                                
                                # Crisp white/light palette matching the Streamlit dashboard
                                fig2.patch.set_facecolor("#ffffff")
                                ax.set_facecolor("#f8fafc")
                                ax.tick_params(colors="#334155")
                                ax.xaxis.label.set_color("#334155")
                                ax.yaxis.label.set_color("#334155")
                                ax.title.set_color("#4f46e5")
                                
                                figures[f"{name} — ROC Curve"] = fig_to_base64(fig2)
                                plt.close(fig2)

                        except Exception as e:
                            st.warning(f"Could not generate figures for {name}: {e}")

            # Generate HTML report
            html = generate_html_report(
                title=title,
                dataset_info=dataset_info,
                model_results=model_results,
                comparison_table_html=comparison_html,
                figures=figures,
                ablation_table_html=ablation_html,
                cryptographic_signatures=cryptographic_signatures,
            )

            st.success("🎉 Report compiled successfully with active cryptographic signatures!")

            # Download button
            st.download_button(
                label="⬇️ Download Cryptographically Stamped HTML Report",
                data=html,
                file_name="nids_evaluation_dissertation_report.html",
                mime="text/html",
                use_container_width=True,
            )

            # Preview Section
            section_header("Report Preview")
            with st.expander("Expand Interactive Live Preview"):
                st.components.v1.html(html, height=800, scrolling=True)
