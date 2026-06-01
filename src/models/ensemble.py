"""
ensemble.py — Hybrid ensemble combining IF, XGBoost, and LSTM-AE scores.

This is the dissertation's *novelty contribution*:
  - Normalize each model's anomaly score to [0, 1]
  - Weighted average fusion with configurable weights
  - Final threshold on the fused score
"""

import numpy as np
from src.utils import get_logger

logger = get_logger("ensemble")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def ensemble_predict(
    if_scores: np.ndarray,
    xgb_scores: np.ndarray,
    ae_scores: np.ndarray,
    weights: tuple = (0.25, 0.40, 0.35),
    threshold: float = 0.5,
):
    """
    Fuse anomaly scores from three models via weighted average.

    Args:
        if_scores:  Isolation Forest anomaly scores (higher = more anomalous)
        xgb_scores: XGBoost attack probabilities [0, 1]
        ae_scores:  LSTM-AE reconstruction errors (higher = more anomalous)
        weights:    (w_if, w_xgb, w_ae) — must sum to 1.0
        threshold:  decision boundary on the fused score

    Returns:
        (binary_predictions, fused_scores)
    """
    w_if, w_xgb, w_ae = weights
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1.0, got {sum(weights)}"

    # Normalize each model's scores to [0, 1]
    if_norm = normalize_scores(if_scores)
    xgb_norm = normalize_scores(xgb_scores)
    ae_norm = normalize_scores(ae_scores)

    # Weighted fusion
    fused = w_if * if_norm + w_xgb * xgb_norm + w_ae * ae_norm

    preds = (fused > threshold).astype(int)

    logger.info(
        f"Ensemble fusion: weights=({w_if:.2f}, {w_xgb:.2f}, {w_ae:.2f}), "
        f"threshold={threshold:.2f}, anomalies={preds.sum()}/{len(preds)}"
    )
    return preds, fused


def find_optimal_threshold(fused_scores: np.ndarray, y_true: np.ndarray):
    """
    Search for the fused-score threshold that maximizes F1 on the validation set.
    Returns (best_threshold, best_f1).
    """
    from sklearn.metrics import f1_score

    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (fused_scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    logger.info(f"Optimal ensemble threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    return best_thresh, best_f1
