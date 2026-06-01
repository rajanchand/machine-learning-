"""
feature_engineering.py — Feature selection, correlation analysis, and PCA.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from src.utils import get_logger, timer

logger = get_logger("feature_engineering")


@timer
def remove_correlated_features(X: np.ndarray, feature_names: list, threshold: float = 0.95):
    """
    Drop features with Pearson correlation above threshold.
    Returns (filtered_X, remaining_feature_names, dropped_features).
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df_filtered = df.drop(columns=to_drop)
    logger.info(f"Removed {len(to_drop)} correlated features (>{threshold}): {to_drop[:10]}...")
    return df_filtered.values, df_filtered.columns.tolist(), to_drop


@timer
def compute_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: list, top_k: int = 20):
    """
    Rank features by mutual information with the target.
    Returns sorted list of (feature_name, mi_score).
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    rankings = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)
    logger.info(f"Top-{top_k} features by MI: {[r[0] for r in rankings[:top_k]]}")
    return rankings


@timer
def apply_pca(X: np.ndarray, n_components: float = 0.95):
    """
    Apply PCA, keeping enough components to explain `n_components` variance.
    Returns (transformed_X, pca_model, explained_variance_ratios).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    logger.info(
        f"PCA: {X.shape[1]} features → {X_pca.shape[1]} components "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})"
    )
    return X_pca, pca, pca.explained_variance_ratio_


def get_feature_importance_df(feature_names: list, importances: np.ndarray) -> pd.DataFrame:
    """Create a sorted DataFrame of feature importances."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
