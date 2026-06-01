"""
one_class_svm.py — One-Class SVM baseline model for unsupervised NIDS.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from src.utils import get_logger, timer, save_sklearn_model

logger = get_logger("one_class_svm")


@timer
def train_one_class_svm(
    X_train_normal: np.ndarray,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> OneClassSVM:
    """
    Train a One-Class SVM on normal traffic only.
    nu matches target contamination rate.
    """
    logger.info(f"Training One-Class SVM: nu={nu}, kernel={kernel}, gamma={gamma}...")
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, cache_size=1000)
    model.fit(X_train_normal)
    logger.info(f"One-Class SVM trained: samples={X_train_normal.shape[0]}")
    return model


def predict_one_class_svm(model: OneClassSVM, X: np.ndarray):
    """
    Predict anomalies using One-Class SVM.
    Returns:
        binary_predictions (0 = normal, 1 = attack)
        anomaly_scores (higher = more anomalous)
    """
    # sklearn returns -1 for anomaly, 1 for normal
    raw_preds = model.predict(X)
    preds = np.where(raw_preds == -1, 1, 0)
    
    # Distance to decision boundary (negated so higher score = more anomalous)
    scores = -model.decision_function(X)
    return preds, scores


def train_and_save(X_train_normal, nu=0.05, kernel="rbf", gamma="scale"):
    """Train and securely save a One-Class SVM model with metadata."""
    model = train_one_class_svm(X_train_normal, nu=nu, kernel=kernel, gamma=gamma)
    metadata = {
        "model_type": "One-Class_SVM",
        "nu": nu,
        "kernel": kernel,
        "gamma": gamma,
        "n_features": X_train_normal.shape[1],
    }
    save_sklearn_model(model, "one_class_svm", metadata)
    return model, metadata
