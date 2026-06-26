from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_score) -> dict:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = []
    for threshold in thresholds:
        scores.append(f1_score(y_true, y_score >= threshold, zero_division=0))

    best_idx = int(np.argmax(scores))
    return {
        "classification_threshold": float(thresholds[best_idx]),
        "f1": float(scores[best_idx]),
    }
