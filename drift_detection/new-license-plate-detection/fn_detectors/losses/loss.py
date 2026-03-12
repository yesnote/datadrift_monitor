import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_classifier(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return accuracy_score(y_true, y_score >= 0.5), roc_auc_score(y_true, y_score)

