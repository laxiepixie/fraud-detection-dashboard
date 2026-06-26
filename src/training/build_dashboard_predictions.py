from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.model_loader import load_artifacts
from src.inference.preprocessing import transform_features


RAW_TEST = ROOT / "data" / "raw" / "fraudTest.csv"
PROJECT_PROCESSED_DIR = ROOT / "data" / "processed"
TEMP_PROCESSED_DIR = Path(tempfile.gettempdir()) / "fraud_detection_processed"


def writable_dir(project_dir: Path, temp_dir: Path) -> Path:
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        test_file = project_dir / ".write_test"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
        return project_dir
    except OSError:
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir


def mahalanobis_score(X: pd.DataFrame, artifact: dict) -> np.ndarray:
    diff = X.to_numpy() - artifact["mean"]
    return np.sqrt(np.sum(diff @ artifact["inv_cov"] * diff, axis=1))


def main() -> None:
    if not RAW_TEST.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {RAW_TEST}")

    print("Membaca fraudTest.csv...")
    df = pd.read_csv(RAW_TEST)

    print("Memuat model...")
    artifacts = load_artifacts()

    print("Membuat fitur...")
    if artifacts.get("preprocessor") is not None:
        X = artifacts["preprocessor"].transform_features(df, artifacts["encoder"], artifacts["scaler"])
    else:
        X = transform_features(df, artifacts["encoder"], artifacts["scaler"])

    print("Membuat prediksi tiap model...")
    threshold = float(artifacts["threshold"].get("classification_threshold", 0.5))
    result = df.copy()

    result["score_lgbm"] = artifacts["classifier"].predict_proba(X)[:, 1]
    result["pred_lgbm"] = (result["score_lgbm"] >= threshold).astype(int)

    result["score_if"] = -artifacts["isolation_forest"].decision_function(X)
    if_threshold = artifacts["threshold"].get("anomaly_thresholds", {}).get("isolation_forest", {}).get("threshold", 0.0)
    result["pred_if"] = (result["score_if"] >= if_threshold).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        result["score_lof"] = -artifacts["lof"].decision_function(X)
        # SVM scoring: prefer decision_function, fallback to predict_proba if needed
        try:
            result["score_svm"] = artifacts["svm"].decision_function(X)
        except Exception:
            if hasattr(artifacts["svm"], "predict_proba"):
                result["score_svm"] = artifacts["svm"].predict_proba(X)[:, 1]
            else:
                result["score_svm"] = np.zeros(len(X))

    lof_threshold = artifacts.get("threshold", {}).get("anomaly_thresholds", {}).get("lof", {}).get("threshold", 0.0)
    result["pred_lof"] = (result["score_lof"] >= lof_threshold).astype(int)

    svm_threshold = artifacts.get("threshold", {}).get("anomaly_thresholds", {}).get("svm", {}).get("threshold", 0.0)
    result["pred_svm"] = (result["score_svm"] >= svm_threshold).astype(int)

    result["score_cpp"] = mahalanobis_score(X, artifacts["mahalanobis"])
    cpp_threshold = artifacts.get("threshold", {}).get("anomaly_thresholds", {}).get("mahalanobis", {}).get("threshold")
    if cpp_threshold is None:
        cpp_threshold = float(np.quantile(result["score_cpp"], 0.99))
    result["pred_cpp"] = (result["score_cpp"] >= cpp_threshold).astype(int)

    output_dir = writable_dir(PROJECT_PROCESSED_DIR, TEMP_PROCESSED_DIR)
    output_path = output_dir / "dashboard_predictions.csv"
    result.to_csv(output_path, index=False)

    print("Selesai.")
    print(f"Output: {os.path.abspath(output_path)}")
    print(
        result[["pred_cpp", "pred_lgbm", "pred_if", "pred_lof", "pred_svm"]]
        .sum()
        .rename("jumlah_prediksi_fraud")
    )


if __name__ == "__main__":
    main()
