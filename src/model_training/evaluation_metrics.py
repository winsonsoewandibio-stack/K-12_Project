"""
evaluation_metrics.py
----------------------
Evaluation metrics for classification models.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Computes standard classification metrics.
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        if y_proba is not None and y_proba.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics

