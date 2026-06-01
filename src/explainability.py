"""
explainability.py — SHAP-based model explanations for feature importance.
"""

import numpy as np
import shap
from src.utils import get_logger

logger = get_logger("explainability")


def explain_xgboost(model, X_sample: np.ndarray, feature_names: list):
    """
    Compute SHAP values for an XGBoost model using TreeExplainer (fast).
    Returns (shap_values, explainer).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info(f"SHAP TreeExplainer computed for {X_sample.shape[0]} samples")
    return shap_values, explainer


def explain_isolation_forest(model, X_sample: np.ndarray, feature_names: list, n_background: int = 100):
    """
    Compute SHAP values for Isolation Forest using KernelExplainer.
    Uses a small background sample for computational feasibility.
    """
    # Use a subset as background
    bg = X_sample[:n_background] if len(X_sample) > n_background else X_sample

    def predict_fn(X):
        return -model.score_samples(X)

    explainer = shap.KernelExplainer(predict_fn, bg)
    # Explain a smaller sample to keep it tractable
    explain_sample = X_sample[:min(50, len(X_sample))]
    shap_values = explainer.shap_values(explain_sample, nsamples=100)
    logger.info(f"SHAP KernelExplainer computed for {explain_sample.shape[0]} samples")
    return shap_values, explainer


def get_top_features(shap_values: np.ndarray, feature_names: list, top_k: int = 15):
    """
    Return the top-k most important features based on mean absolute SHAP values.
    Returns list of (feature_name, mean_abs_shap).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_k]
    result = [(feature_names[i], float(mean_abs[i])) for i in indices]
    return result
