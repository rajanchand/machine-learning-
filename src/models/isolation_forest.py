"""
isolation_forest.py — Isolation Forest baseline for unsupervised anomaly detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from src.utils import get_logger, timer, save_sklearn_model

logger = get_logger("isolation_forest")


@timer
def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 200,
    max_samples: str = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest on the training data.
    For unsupervised anomaly detection: trained on ALL data (normal + anomalous).
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info(
        f"Isolation Forest trained: n_estimators={n_estimators}, "
        f"contamination={contamination}, samples={X_train.shape[0]}"
    )
    return model


def predict_isolation_forest(model: IsolationForest, X: np.ndarray):
    """
    Return anomaly scores and binary predictions.
    Scores are inverted so higher = more anomalous (consistent with other models).
    """
    # score_samples: lower (more negative) = more anomalous
    raw_scores = -model.score_samples(X)  # invert so higher = anomaly
    # predict: -1 = anomaly, 1 = normal → convert to 0/1
    preds = (model.predict(X) == -1).astype(int)
    return preds, raw_scores


def train_and_save(X_train, y_train=None, **kwargs):
    """Train, evaluate informally, and save the Isolation Forest model."""
    model = train_isolation_forest(X_train, **kwargs)
    metadata = {
        "model_type": "IsolationForest",
        "n_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "contamination": kwargs.get("contamination", 0.01),
        "n_estimators": kwargs.get("n_estimators", 200),
    }
    save_sklearn_model(model, "isolation_forest", metadata)
    return model, metadata
