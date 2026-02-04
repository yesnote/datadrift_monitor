import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


def compute_binary_classification_metrics(
    scores,
    labels,
    compute_roc=True,
):
    """
    Args:
        scores (list or np.ndarray):
            Continuous scores (e.g., DiL). Higher = more likely positive.
        labels (list or np.ndarray):
            Binary GT labels.
            1: missed detection
            0: normal detection
        compute_roc (bool):
            Whether to also compute ROC metrics.

    Returns:
        metrics (dict):
            {
              "pr": {
                  "precision": np.ndarray,
                  "recall": np.ndarray,
                  "thresholds": np.ndarray,
                  "ap": float,
              },
              "roc": {           # only if compute_roc=True
                  "fpr": np.ndarray,
                  "tpr": np.ndarray,
                  "thresholds": np.ndarray,
                  "auc": float,
              }
            }
    """
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)

    # Precision–Recall
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    metrics = {
        "pr": {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
            "ap": ap,
        }
    }

    # ROC (optional)
    if compute_roc:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        metrics["roc"] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
            "auc": roc_auc,
        }

    return metrics
