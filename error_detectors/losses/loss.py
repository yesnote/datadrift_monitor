from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_classifier(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
