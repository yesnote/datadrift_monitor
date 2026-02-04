import numpy as np

def eval_thresholds(dils, missed, thresholds):
    out = []
    for t in thresholds:
        pred = [d >= t for d in dils]
        tp = sum(p and m for p, m in zip(pred, missed))
        fp = sum(p and not m for p, m in zip(pred, missed))
        fn = sum((not p) and m for p, m in zip(pred, missed))

        recall = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        out.append((t, recall, precision))
    return out
