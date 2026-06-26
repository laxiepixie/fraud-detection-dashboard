from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.kernel_approximation import RBFSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.preprocessing import add_features, fit_transform_features, transform_features
from src.training.evaluate_model import evaluate_classifier
from src.training.optimize_threshold import find_best_threshold

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYTICS_DIR = ROOT / "data" / "analytics"
MODEL_DIR = ROOT / "models"
ANOMALY_SAMPLE_SIZE = 100_000

# Use temp directory for model storage if models directory is not writable
_model_dir_to_use = None

def get_model_dir():
    global _model_dir_to_use
    if _model_dir_to_use is None:
        # Try to write to the main models directory
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            test_file = MODEL_DIR / ".write_test"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            _model_dir_to_use = MODEL_DIR
        except (PermissionError, OSError):
            # Fall back to temp directory
            temp_models_dir = Path(tempfile.gettempdir()) / "fraud_detection_models"
            temp_models_dir.mkdir(parents=True, exist_ok=True)
            _model_dir_to_use = temp_models_dir
            print(f"Warning: Using temp directory for models: {_model_dir_to_use}")
    return _model_dir_to_use


_processed_dir_to_use = None

def get_processed_dir():
    global _processed_dir_to_use
    if _processed_dir_to_use is None:
        # Try to write to the main processed directory
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            test_file = PROCESSED_DIR / ".write_test"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            _processed_dir_to_use = PROCESSED_DIR
        except (PermissionError, OSError):
            # Fall back to temp directory
            temp_processed_dir = Path(tempfile.gettempdir()) / "fraud_detection_processed"
            temp_processed_dir.mkdir(parents=True, exist_ok=True)
            _processed_dir_to_use = temp_processed_dir
            print(f"Warning: Using temp directory for processed data: {_processed_dir_to_use}")
    return _processed_dir_to_use


_analytics_dir_to_use = None

def get_analytics_dir():
    global _analytics_dir_to_use
    if _analytics_dir_to_use is None:
        # Try to write to the main analytics directory
        try:
            ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
            test_file = ANALYTICS_DIR / ".write_test"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            _analytics_dir_to_use = ANALYTICS_DIR
        except (PermissionError, OSError):
            # Fall back to temp directory
            temp_analytics_dir = Path(tempfile.gettempdir()) / "fraud_detection_analytics"
            temp_analytics_dir.mkdir(parents=True, exist_ok=True)
            _analytics_dir_to_use = temp_analytics_dir
            print(f"Warning: Using temp directory for analytics data: {_analytics_dir_to_use}")
    return _analytics_dir_to_use


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, ANALYTICS_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output_dirs_writable() -> None:
    # Just initialize the fallback directories - they will be created if needed
    get_model_dir()
    get_processed_dir()
    get_analytics_dir()
    # No need to check further - we have fallbacks


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = RAW_DIR / "fraudTrain.csv"
    test_path = RAW_DIR / "fraudTest.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Letakkan fraudTrain.csv dan fraudTest.csv di data/raw/")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_classifier(scale_pos_weight: float):
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            verbosity=-1,
        )
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def save_analytics(train_df: pd.DataFrame) -> None:
    analytics_save_dir = get_analytics_dir()
    enriched = add_features(train_df)
    enriched.groupby("state")["is_fraud"].agg(["count", "sum", "mean"]).reset_index().to_csv(
        os.path.join(analytics_save_dir, "fraud_by_state.csv"), index=False
    )
    enriched.groupby("category")["is_fraud"].agg(["count", "sum", "mean"]).reset_index().to_csv(
        os.path.join(analytics_save_dir, "fraud_by_category.csv"), index=False
    )
    enriched.groupby("hour")["is_fraud"].agg(["count", "sum", "mean"]).reset_index().to_csv(
        os.path.join(analytics_save_dir, "fraud_by_hour.csv"), index=False
    )
    enriched.groupby("gender")["is_fraud"].agg(["count", "sum", "mean"]).reset_index().to_csv(
        os.path.join(analytics_save_dir, "fraud_by_gender.csv"), index=False
    )


def mahalanobis_scores(X: pd.DataFrame, artifact: dict) -> np.ndarray:
    diff = X.to_numpy() - artifact["mean"]
    return np.sqrt(np.sum(diff @ artifact["inv_cov"] * diff, axis=1))


def optimize_score_threshold(y_true, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    scores = scores[finite]
    y_true = y_true[finite]
    if scores.size == 0:
        return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.999, 250)))
    best = {"threshold": float(candidates[0]), "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if (f1, recall) > (best["f1"], best["recall"]):
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def build_dashboard_predictions(
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    classifier,
    isolation_forest,
    lof,
    svm,
    mahalanobis: dict,
    threshold: dict,
    normal_train: pd.DataFrame,
) -> pd.DataFrame:
    result = test_df.copy()

    lgbm_probability = classifier.predict_proba(X_test)[:, 1]
    result["score_lgbm"] = lgbm_probability
    result["pred_lgbm"] = (lgbm_probability >= threshold["classification_threshold"]).astype(int)

    result["score_if"] = -isolation_forest.decision_function(X_test)
    if_threshold = threshold.get("anomaly_thresholds", {}).get("isolation_forest", {}).get("threshold", 0.0)
    result["pred_if"] = (result["score_if"] >= if_threshold).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        result["score_lof"] = -lof.decision_function(X_test)
    lof_threshold = threshold.get("anomaly_thresholds", {}).get("lof", {}).get("threshold", 0.0)
    result["pred_lof"] = (result["score_lof"] >= lof_threshold).astype(int)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        result["score_svm"] = -svm.decision_function(X_test)
    svm_threshold = threshold.get("anomaly_thresholds", {}).get("svm", {}).get("threshold", 0.0)
    result["pred_svm"] = (result["score_svm"] >= svm_threshold).astype(int)
    result["score_cpp"] = mahalanobis_scores(X_test, mahalanobis)
    
    cpp_threshold = threshold.get("anomaly_thresholds", {}).get("mahalanobis", {}).get("threshold")
    if cpp_threshold is None:
        train_maha_score = mahalanobis_scores(normal_train, mahalanobis)
        cpp_threshold = float(np.quantile(train_maha_score, 0.99))
    result["pred_cpp"] = (result["score_cpp"] >= cpp_threshold).astype(int)

    return result


def _copy_files_to_project_dirs() -> None:
    """Try to copy files from temp directories to project directories if possible."""
    def can_write_to(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".copy_write_test"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    if not can_write_to(MODEL_DIR) or not can_write_to(PROCESSED_DIR) or not can_write_to(ANALYTICS_DIR):
        print("  Folder project masih tidak bisa ditulis, jadi file tetap dipakai dari temp directory.")
        print(f"  Models: {Path(tempfile.gettempdir()) / 'fraud_detection_models'}")
        print(f"  Processed: {Path(tempfile.gettempdir()) / 'fraud_detection_processed'}")
        print(f"  Analytics: {Path(tempfile.gettempdir()) / 'fraud_detection_analytics'}")
        return

    # Try to copy models
    try:
        src_models = Path(tempfile.gettempdir()) / "fraud_detection_models"
        if src_models.exists():
            dest_models = MODEL_DIR
            dest_models.mkdir(parents=True, exist_ok=True)
            for file in src_models.glob("*"):
                shutil.copy2(file, dest_models / file.name)
            print(f"✓ Models copied to: {dest_models}")
    except Exception as e:
        print(f"  (Could not copy models to project: {e})")
    
    # Try to copy processed data
    try:
        src_processed = Path(tempfile.gettempdir()) / "fraud_detection_processed"
        if src_processed.exists():
            dest_processed = PROCESSED_DIR
            dest_processed.mkdir(parents=True, exist_ok=True)
            for file in src_processed.glob("*"):
                shutil.copy2(file, dest_processed / file.name)
            print(f"✓ Processed data copied to: {dest_processed}")
    except Exception as e:
        print(f"  (Could not copy processed data: {e})")
    
    # Try to copy analytics
    try:
        src_analytics = Path(tempfile.gettempdir()) / "fraud_detection_analytics"
        if src_analytics.exists():
            dest_analytics = ANALYTICS_DIR
            dest_analytics.mkdir(parents=True, exist_ok=True)
            for file in src_analytics.glob("*"):
                shutil.copy2(file, dest_analytics / file.name)
            print(f"✓ Analytics data copied to: {dest_analytics}")
    except Exception as e:
        print(f"  (Could not copy analytics: {e})")


def main() -> None:
    assert_output_dirs_writable()
    print("Membaca dataset...")
    train_df, test_df = load_data()

    if "is_fraud" not in train_df.columns:
        raise ValueError("Kolom target `is_fraud` tidak ditemukan di fraudTrain.csv")

    print("Membuat fitur training...")
    X, artifacts = fit_transform_features(train_df)
    y = train_df["is_fraud"].astype(int)
    print("Membuat fitur testing...")
    X_test = artifacts.preprocessor.transform_features(test_df, artifacts.encoder, artifacts.scaler)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    fraud_count = max(int(y_train.sum()), 1)
    non_fraud_count = max(int((y_train == 0).sum()), 1)
    classifier = build_classifier(scale_pos_weight=non_fraud_count / fraud_count)
    print("Melatih model klasifikasi fraud...")
    classifier.fit(X_train, y_train)

    print("Mengoptimalkan threshold...")
    valid_score = classifier.predict_proba(X_valid)[:, 1]
    threshold = find_best_threshold(y_valid, valid_score)
    metrics = evaluate_classifier(y_valid, valid_score, threshold["classification_threshold"])

    normal_train = X_train[y_train == 0]
    if len(normal_train) > ANOMALY_SAMPLE_SIZE:
        print(f"Mengambil sample {ANOMALY_SAMPLE_SIZE:,} transaksi normal untuk anomaly models...")
        normal_train = normal_train.sample(ANOMALY_SAMPLE_SIZE, random_state=42)

    print("Melatih Isolation Forest...")
    isolation_forest = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    isolation_forest.fit(normal_train)

    print("Melatih Local Outlier Factor...")
    lof = Pipeline([
        ("scaler", StandardScaler()),
        (
            "lof",
            LocalOutlierFactor(n_neighbors=25, novelty=True, contamination="auto", n_jobs=-1),
        ),
    ])
    lof.fit(normal_train)

    print("Melatih Support Vector Machine (One-Class SVM)...")
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)),
    ])
    # Penyelamat RAM: Batasi maksimal 15.000 baris khusus untuk SVM
    svm_train_sample = normal_train.sample(min(15000, len(normal_train)), random_state=42)
    svm.fit(svm_train_sample)

    print("Menghitung Mahalanobis artifact...")
    cov = LedoitWolf().fit(normal_train)
    mahalanobis = {"mean": cov.location_, "inv_cov": np.linalg.pinv(cov.covariance_)}

    print("Mengoptimalkan threshold anomaly models...")
    valid_if_score = -isolation_forest.decision_function(X_valid)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        valid_lof_score = -lof.decision_function(X_valid)
        valid_svm_score = -svm.decision_function(X_valid)
    valid_maha_score = mahalanobis_scores(X_valid, mahalanobis)
    anomaly_thresholds = {
        "isolation_forest": optimize_score_threshold(y_valid, valid_if_score),
        "lof": optimize_score_threshold(y_valid, valid_lof_score),
        "svm": optimize_score_threshold(y_valid, valid_svm_score),
        "mahalanobis": optimize_score_threshold(y_valid, valid_maha_score),
    }
    for name, info in anomaly_thresholds.items():
        print(
            f"{name} threshold={info['threshold']:.6f} "
            f"precision={info['precision']:.4f} recall={info['recall']:.4f} f1={info['f1']:.4f}"
        )
    threshold = {**threshold, "anomaly_thresholds": anomaly_thresholds}

    print("Menyimpan model...")
    ensure_dirs()
    
    # Get the appropriate model directory (may fall back to temp)
    model_save_dir = get_model_dir()
    model_dir_str = os.path.abspath(str(model_save_dir))
    
    joblib.dump(classifier, os.path.join(model_dir_str, "lightgbm.pkl"))
    joblib.dump(isolation_forest, os.path.join(model_dir_str, "isolation_forest.pkl"))
    joblib.dump(lof, os.path.join(model_dir_str, "lof.pkl"))
    joblib.dump(svm, os.path.join(model_dir_str, "svm_model.pkl"))
    joblib.dump(mahalanobis, os.path.join(model_dir_str, "mahalanobis.pkl"))
    joblib.dump(artifacts.encoder, os.path.join(model_dir_str, "encoder.pkl"))
    joblib.dump(artifacts.scaler, os.path.join(model_dir_str, "scaler.pkl"))
    joblib.dump(artifacts.preprocessor, os.path.join(model_dir_str, "preprocessor.pkl"))

    with open(os.path.join(model_dir_str, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({**threshold, "metrics": metrics}, f, indent=2)

    print("Menyimpan fitur dan analytics...")
    ensure_dirs()
    processed_save_dir = get_processed_dir()
    processed_dir_str = os.path.abspath(str(processed_save_dir))
    X.to_parquet(os.path.join(processed_dir_str, "train_features.parquet"), index=False)
    X_test.to_parquet(os.path.join(processed_dir_str, "test_features.parquet"), index=False)
    print("Menyimpan dashboard_predictions.csv...")
    dashboard_df = build_dashboard_predictions(
        test_df,
        X_test,
        classifier,
        isolation_forest,
        lof,
        svm,
        mahalanobis,
        threshold,
        normal_train,
    )
    dashboard_df.to_csv(os.path.join(processed_dir_str, "dashboard_predictions.csv"), index=False)
    save_analytics(train_df)

    print("Training selesai.")
    print(f"Best threshold: {threshold['classification_threshold']:.2f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Try to copy files to project directories
    print("\nMenyalin file ke folder project...")
    _copy_files_to_project_dirs()


if __name__ == "__main__":
    main()
