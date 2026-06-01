"""
evaluation.py — Evaluation metrics, confusion matrix, ROC curves, and ablation helpers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from src.utils import get_logger

logger = get_logger("evaluation")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    """
    Compute a full set of evaluation metrics.
    Returns a dict suitable for display and report generation.
    """
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["FPR"] = round(fp / max(fp + tn, 1), 4)
    metrics["FNR"] = round(fn / max(fn + tp, 1), 4)
    metrics["TP"] = int(tp)
    metrics["FP"] = int(fp)
    metrics["TN"] = int(tn)
    metrics["FN"] = int(fn)

    # ROC-AUC (requires continuous scores)
    if y_scores is not None:
        try:
            metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_scores), 4)
        except ValueError:
            metrics["ROC-AUC"] = None
    else:
        metrics["ROC-AUC"] = None

    return metrics


def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as numpy array."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_roc_curve(y_true, y_scores):
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_scores)


def get_pr_curve(y_true, y_scores):
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_scores)


def get_classification_report(y_true, y_pred) -> str:
    """Return a formatted classification report string."""
    return classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"], zero_division=0
    )


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    Args:
        results: dict mapping model_name → metrics_dict
    Returns:
        DataFrame with models as rows, metrics as columns.
    """
    rows = []
    for model_name, metrics_dict in results.items():
        row = {"Model": model_name}
        row.update(metrics_dict)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df


def ablation_study(base_metrics: dict, variants: dict) -> pd.DataFrame:
    """
    Generate an ablation study table comparing a base model with variants.
    Args:
        base_metrics: metrics dict for the full model
        variants: dict mapping variant_name → metrics_dict
    Returns:
        DataFrame with diff columns showing impact of each ablation.
    """
    rows = [{"Variant": "Full Model (Proposed)", **base_metrics}]
    for name, v_metrics in variants.items():
        row = {"Variant": name}
        for key in base_metrics:
            if isinstance(base_metrics[key], (int, float)) and base_metrics[key] is not None:
                val = v_metrics.get(key)
                if val is not None:
                    row[key] = val
                    row[f"{key}_Δ"] = round(val - base_metrics[key], 4)
                else:
                    row[key] = None
                    row[f"{key}_Δ"] = None
            else:
                row[key] = v_metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variant")
