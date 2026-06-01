"""
charts.py — Reusable chart components for the Streamlit app.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation import get_confusion_matrix, get_roc_curve, get_pr_curve


PLOTLY_TEMPLATE = "plotly_white"
COLORS = {
    "primary": "#4f46e5",
    "secondary": "#6366f1",
    "success": "#16a34a",
    "danger": "#dc2626",
    "warning": "#ca8a04",
    "info": "#2563eb",
    "normal": "#16a34a",
    "attack": "#dc2626",
}


def plot_class_distribution(y, title="Class Distribution"):
    """Bar chart showing normal vs attack distribution."""
    normal = int((y == 0).sum())
    attack = int((y == 1).sum())
    fig = go.Figure(data=[
        go.Bar(
            x=["Normal", "Attack"],
            y=[normal, attack],
            marker_color=[COLORS["normal"], COLORS["attack"]],
            text=[f"{normal:,}", f"{attack:,}"],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE,
        height=350, yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Heatmap confusion matrix using matplotlib."""
    cm = get_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.tick_params(colors="#333333")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#4f46e5")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_scores, model_name="Model"):
    """Interactive ROC curve using Plotly."""
    fpr, tpr, _ = get_roc_curve(y_true, y_scores)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"{model_name} (AUC={roc_auc:.4f})",
        line=dict(color=COLORS["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="#555", dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve", template=PLOTLY_TEMPLATE,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_pr_curve(y_true, y_scores, model_name="Model"):
    """Interactive Precision-Recall curve."""
    precision, recall, _ = get_pr_curve(y_true, y_scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=model_name,
        line=dict(color=COLORS["secondary"], width=2),
    ))
    fig.update_layout(
        title="Precision-Recall Curve", template=PLOTLY_TEMPLATE,
        xaxis_title="Recall", yaxis_title="Precision",
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_anomaly_score_distribution(scores, y_true=None, threshold=None, title="Anomaly Score Distribution"):
    """Histogram of anomaly scores, optionally colored by true label."""
    fig = go.Figure()
    if y_true is not None:
        fig.add_trace(go.Histogram(
            x=scores[y_true == 0], name="Normal",
            marker_color=COLORS["normal"], opacity=0.7, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=scores[y_true == 1], name="Attack",
            marker_color=COLORS["attack"], opacity=0.7, nbinsx=50,
        ))
    else:
        fig.add_trace(go.Histogram(
            x=scores, name="Scores",
            marker_color=COLORS["primary"], opacity=0.8, nbinsx=50,
        ))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"Threshold: {threshold:.4f}")
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, barmode="overlay",
        xaxis_title="Anomaly Score", yaxis_title="Count",
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_training_loss(train_losses, val_losses=None, title="Training Loss"):
    """Line plot of training and validation loss over epochs."""
    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses, mode="lines",
        name="Train Loss", line=dict(color=COLORS["primary"], width=2),
    ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_losses, mode="lines",
            name="Val Loss", line=dict(color=COLORS["warning"], width=2),
        ))
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE,
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_model_comparison_radar(results_dict):
    """Radar chart comparing models across key metrics."""
    metrics_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    colors = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["info"]]
    for i, (name, metrics) in enumerate(results_dict.items()):
        values = [metrics.get(k, 0) or 0 for k in metrics_keys]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values, theta=metrics_keys + [metrics_keys[0]],
            fill="toself", name=name,
            line_color=colors[i % len(colors)], opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=PLOTLY_TEMPLATE, height=500,
        title="Model Comparison — Radar Chart",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_feature_importance(feature_names, importances, top_k=15, title="Feature Importance"):
    """Horizontal bar chart of top feature importances."""
    # Sort and take top k
    indices = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in indices]
    vals = [importances[i] for i in indices]

    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1], orientation="h",
        marker_color=COLORS["secondary"],
    ))
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE,
        xaxis_title="Importance", height=max(350, top_k * 28),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
