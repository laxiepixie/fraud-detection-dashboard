from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.inference.storage import PROJECT_MODEL_DIR, TEMP_MODEL_DIR

MODEL_DIR = PROJECT_MODEL_DIR


REQUIRED_MODEL_FILES = [
    "lightgbm.pkl",
    "isolation_forest.pkl",
    "lof.pkl",
    "mahalanobis.pkl",
    "svm_model.pkl",
    "encoder.pkl",
    "scaler.pkl",
    "preprocessor.pkl",
    "threshold.json",
]


def _has_all_model_files(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_MODEL_FILES)


def _get_model_dir() -> Path:
    """Get model directory, falling back to temp if project directory doesn't have models."""
    if _has_all_model_files(MODEL_DIR):
        return MODEL_DIR

    if _has_all_model_files(TEMP_MODEL_DIR):
        print(f"Loading models from temp directory: {TEMP_MODEL_DIR}")
        return TEMP_MODEL_DIR

    return MODEL_DIR


def load_artifacts(model_dir: Path | None = None) -> dict:
    if model_dir is None or not _has_all_model_files(model_dir):
        model_dir = _get_model_dir()

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Model belum lengkap: {', '.join(missing)}. Jalankan training dulu.")

    with open(model_dir / "threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)

    return {
        "classifier": joblib.load(model_dir / "lightgbm.pkl"),
        "isolation_forest": joblib.load(model_dir / "isolation_forest.pkl"),
        "lof": joblib.load(model_dir / "lof.pkl"),
        "mahalanobis": joblib.load(model_dir / "mahalanobis.pkl"),
        "svm": joblib.load(model_dir / "svm_model.pkl"),
        "encoder": joblib.load(model_dir / "encoder.pkl"),
        "scaler": joblib.load(model_dir / "scaler.pkl"),
        "preprocessor": joblib.load(model_dir / "preprocessor.pkl"),
        "threshold": threshold,
        "model_dir": model_dir,
    }
