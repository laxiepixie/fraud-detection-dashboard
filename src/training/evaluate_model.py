from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_classifier(y_true, y_score, threshold: float) -> dict:
    y_pred = y_score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
