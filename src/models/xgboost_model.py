"""
xgboost_model.py — XGBoost supervised classifier for network intrusion detection.
"""

import numpy as np
import xgboost as xgb
from src.utils import get_logger, timer, save_sklearn_model

logger = get_logger("xgboost_model")


@timer
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier.
    Automatically handles class imbalance via scale_pos_weight.
    """
    # Calculate class imbalance ratio
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    scale_pos_weight = n_normal / max(n_attack, 1)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    logger.info(
        f"XGBoost trained: n_estimators={n_estimators}, max_depth={max_depth}, "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )
    return model


def predict_xgboost(model: xgb.XGBClassifier, X: np.ndarray):
    """Return binary predictions and probability scores."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]  # probability of attack class
    return preds, proba


def get_feature_importance(model: xgb.XGBClassifier):
    """Extract feature importance scores from the trained model."""
    return model.feature_importances_


def train_and_save(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Train, evaluate informally, and save the XGBoost model."""
    model = train_xgboost(X_train, y_train, X_val, y_val, **kwargs)
    metadata = {
        "model_type": "XGBoost",
        "n_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "n_estimators": kwargs.get("n_estimators", 300),
        "max_depth": kwargs.get("max_depth", 6),
    }
    save_sklearn_model(model, "xgboost", metadata)
    return model, metadata
