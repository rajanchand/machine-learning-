"""
Page 8 — Report: Generate and download a full HTML evaluation report.
"""

import streamlit as st
from streamlit_app.components.styles import section_header
from streamlit_app.components.charts import plot_confusion_matrix, plot_roc_curve
from src.report_generator import generate_html_report, fig_to_base64
from src.evaluation import compare_models


def render():
    st.markdown("# 📄 Report Generator")
    st.markdown("Generate a comprehensive evaluation report for your dissertation.")
    st.markdown("---")

    results = st.session_state.get("results", {})
    data = st.session_state.get("data")

    if not results or not data:
        st.warning("⚠️ Train models and view results before generating a report.")
        return

    section_header("Report Configuration")

    title = st.text_input("Report Title", "ML-Based Network Anomaly Detection — Evaluation Report")

    models_to_include = st.multiselect(
        "Models to include", list(results.keys()), default=list(results.keys())
    )

    include_figures = st.checkbox("Include visualisations", value=True)

    if st.button("📄 Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            y_test = data["y_test"]

            # Dataset info
            dataset_info = {
                "Format": data["dataset_format"],
                "Features": data["n_features"],
                "Total Samples": sum(data["class_distribution"].values()),
                "Normal Samples": data["class_distribution"]["normal"],
                "Attack Samples": data["class_distribution"]["attack"],
                "Train Size": len(data["y_train"]),
                "Validation Size": len(data["y_val"]),
                "Test Size": len(data["y_test"]),
            }

            # Model results
            model_results = {}
            for name in models_to_include:
                if name in results:
                    model_results[name] = results[name]["metrics"]

            # Comparison table
            comparison_html = ""
            if len(model_results) >= 2:
                df = compare_models(model_results)
                comparison_html = df.to_html(classes="comparison-table")

            # Figures
            figures = {}
            if include_figures:
                for name in models_to_include:
                    if name in results:
                        r = results[name]
                        try:
                            # Confusion matrix
                            fig = plot_confusion_matrix(y_test, r["preds"], f"{name} — Confusion Matrix")
                            figures[f"{name} — Confusion Matrix"] = fig_to_base64(fig)

                            # ROC curve (as matplotlib for embedding)
                            if r["scores"] is not None:
                                import matplotlib.pyplot as plt
                                from sklearn.metrics import roc_curve, auc
                                fpr, tpr, _ = roc_curve(y_test, r["scores"])
                                roc_auc = auc(fpr, tpr)
                                fig2, ax = plt.subplots(figsize=(7, 5))
                                ax.plot(fpr, tpr, color="#7c3aed", lw=2, label=f"AUC={roc_auc:.4f}")
                                ax.plot([0, 1], [0, 1], "--", color="#555")
                                ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                                ax.set_title(f"{name} — ROC Curve")
                                ax.legend()
                                fig2.patch.set_facecolor("#0f0f1a")
                                ax.set_facecolor("#1a1a2e")
                                ax.tick_params(colors="#ccc")
                                ax.xaxis.label.set_color("#ccc")
                                ax.yaxis.label.set_color("#ccc")
                                ax.title.set_color("#c4b5fd")
                                figures[f"{name} — ROC Curve"] = fig_to_base64(fig2)
                                plt.close(fig2)

                        except Exception as e:
                            st.warning(f"Could not generate figures for {name}: {e}")

            # Generate HTML
            html = generate_html_report(
                title=title,
                dataset_info=dataset_info,
                model_results=model_results,
                comparison_table_html=comparison_html,
                figures=figures,
            )

            st.success("✅ Report generated successfully!")

            # Download button
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html,
                file_name="anomaly_detection_report.html",
                mime="text/html",
                use_container_width=True,
            )

            # Preview
            section_header("Report Preview")
            with st.expander("View report preview"):
                st.components.v1.html(html, height=800, scrolling=True)
