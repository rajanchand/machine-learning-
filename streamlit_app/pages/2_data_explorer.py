"""
Page 2 — Data Explorer: Upload, preview, statistics, and class distribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_app.components.styles import section_header, metric_card
from streamlit_app.components.charts import plot_class_distribution
from src.preprocessing import (
    load_csv, load_pcap, detect_dataset_format, clean_data,
    extract_labels, encode_and_scale, split_data, save_preprocessing_artifacts,
)
from src.utils import set_seed


def render():
    st.markdown("# 📊 Data Explorer")
    st.markdown("Upload your network traffic dataset to begin analysis.")
    st.markdown("---")

    # ── File Upload ──
    section_header("Upload Dataset")
    uploaded = st.file_uploader(
        "Choose a CSV or PCAP file",
        type=["csv", "pcap", "pcapng"],
        help="Supports CIC-IDS2017, UNSW-NB15, or any CSV with a label column.",
    )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test split (%)", 5, 30, 15) / 100
    with col2:
        val_size = st.slider("Validation split (%)", 5, 30, 15) / 100

    # ── Demo / Mock Dataset ──
    st.markdown("💡 **No dataset ready?** Load a high-fidelity synthetic demo dataset instantly:")
    demo_cols = st.columns([2, 1, 1])
    with demo_cols[0]:
        demo_format = st.selectbox("Demo Format", ["UNSW-NB15", "CIC-IDS2017"], label_visibility="collapsed")
    with demo_cols[1]:
        demo_samples = st.selectbox("Samples", [1000, 2000, 5000], index=1, label_visibility="collapsed")
    with demo_cols[2]:
        load_demo = st.button("🚀 Load Demo", use_container_width=True)

    df_raw = None
    dataset_format = None

    if uploaded is not None:
        # Save uploaded file temporarily
        import tempfile, os
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Loading and preprocessing dataset..."):
                set_seed(42)

                # Load
                if suffix in [".pcap", ".pcapng"]:
                    df_raw = load_pcap(tmp_path)
                else:
                    df_raw = load_csv(tmp_path)

                dataset_format = detect_dataset_format(df_raw)
        except Exception as e:
            st.error(f"Error loading file: {e}")
        finally:
            os.unlink(tmp_path)
            
    elif load_demo:
        with st.spinner("Generating synthetic high-fidelity demo dataset..."):
            from src.preprocessing import generate_mock_dataset
            df_raw = generate_mock_dataset(demo_format, demo_samples)
            dataset_format = demo_format

    if df_raw is not None:
        try:
            st.success(f"Loaded dataset: **{dataset_format}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

            # ── Raw Data Preview ──
            section_header("Raw Data Preview")
            st.dataframe(df_raw.head(100), use_container_width=True, height=300)

            # ── Cleaning ──
            with st.spinner("Cleaning data..."):
                df_clean = clean_data(df_raw.copy())

            # ── Labels ──
            df_features, labels, label_col = extract_labels(df_clean, dataset_format)
            y = labels.values

            # ── Stats ──
            section_header("Dataset Statistics")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(metric_card("Total Samples", f"{len(y):,}"), unsafe_allow_html=True)
            with s2:
                st.markdown(metric_card("Features", f"{df_features.shape[1]}"), unsafe_allow_html=True)
            with s3:
                n_normal = int((y == 0).sum())
                st.markdown(metric_card("Normal", f"{n_normal:,}"), unsafe_allow_html=True)
            with s4:
                n_attack = int((y == 1).sum())
                st.markdown(metric_card("Attack", f"{n_attack:,}"), unsafe_allow_html=True)

            # Class distribution chart
            st.plotly_chart(plot_class_distribution(y), use_container_width=True)

            # ── Feature Statistics ──
            section_header("Feature Statistics")
            with st.expander("View detailed feature statistics"):
                st.dataframe(df_features.describe().T.round(4), use_container_width=True)

            # ── Missing Values ──
            missing = df_features.isnull().sum()
            if missing.sum() > 0:
                st.warning(f"⚠️ {missing.sum()} missing values found (already cleaned)")

            # ── Encode & Scale ──
            with st.spinner("Encoding and scaling features..."):
                X_scaled, scaler, label_encoders, feature_names = encode_and_scale(df_features.copy())

            # ── Split ──
            with st.spinner("Splitting into train/val/test..."):
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X_scaled, y, test_size=test_size, val_size=val_size
                )

            # Save artifacts
            save_preprocessing_artifacts(scaler, label_encoders, feature_names)

            # Store in session state
            st.session_state["data"] = {
                "X_train": X_train, "X_val": X_val, "X_test": X_test,
                "y_train": y_train, "y_val": y_val, "y_test": y_test,
                "scaler": scaler, "label_encoders": label_encoders,
                "feature_names": feature_names,
                "dataset_format": dataset_format,
                "label_column": label_col,
                "n_features": X_scaled.shape[1],
                "class_distribution": {"normal": n_normal, "attack": n_attack},
            }

            # ── Split Summary ──
            section_header("Data Split Summary")
            split_df = pd.DataFrame({
                "Split": ["Train", "Validation", "Test"],
                "Samples": [len(y_train), len(y_val), len(y_test)],
                "Normal": [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()],
                "Attack": [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()],
            })
            st.dataframe(split_df, use_container_width=True, hide_index=True)

            st.success("✅ Data preprocessed and stored in session. Proceed to **Model Training**.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())

    elif "data" in st.session_state:
        info = st.session_state["data"]
        st.info(f"Dataset already loaded: **{info['dataset_format']}** — {info['n_features']} features")
        section_header("Current Data Split")
        split_df = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Samples": [len(info["y_train"]), len(info["y_val"]), len(info["y_test"])],
            "Normal": [(info["y_train"]==0).sum(), (info["y_val"]==0).sum(), (info["y_test"]==0).sum()],
            "Attack": [(info["y_train"]==1).sum(), (info["y_val"]==1).sum(), (info["y_test"]==1).sum()],
        })
        st.dataframe(split_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Upload a dataset or load a demo dataset above to begin.")

