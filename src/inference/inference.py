from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference.preprocessing import transform_features


def mahalanobis_score(X: pd.DataFrame, mahalanobis_artifact: dict) -> np.ndarray:
    mean = mahalanobis_artifact["mean"]
    inv_cov = mahalanobis_artifact["inv_cov"]
    diff = X.to_numpy() - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


def predict_transactions(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    if artifacts.get("preprocessor") is not None:
        X = artifacts["preprocessor"].transform_features(df, artifacts["encoder"], artifacts["scaler"])
    else:
        X = transform_features(df, artifacts["encoder"], artifacts["scaler"])
    classifier = artifacts["classifier"]
    threshold = float(artifacts["threshold"].get("classification_threshold", 0.5))

    if hasattr(classifier, "predict_proba"):
        fraud_probability = classifier.predict_proba(X)[:, 1]
    else:
        raw = classifier.decision_function(X)
        fraud_probability = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    iso_score = -artifacts["isolation_forest"].decision_function(X)
    lof_score = -artifacts["lof"].decision_function(X)
    maha_score = mahalanobis_score(X, artifacts["mahalanobis"])

    result = df.copy()
    result["fraud_probability"] = fraud_probability
    result["is_predicted_fraud"] = fraud_probability >= threshold
    result["isolation_anomaly_score"] = iso_score
    result["lof_anomaly_score"] = lof_score
    result["mahalanobis_score"] = maha_score
    return result.sort_values("fraud_probability", ascending=False)
