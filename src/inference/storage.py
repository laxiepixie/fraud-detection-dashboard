from __future__ import annotations

import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROJECT_MODEL_DIR = ROOT / "models"
PROJECT_PROCESSED_DIR = ROOT / "data" / "processed"
PROJECT_ANALYTICS_DIR = ROOT / "data" / "analytics"

TEMP_MODEL_DIR = Path(tempfile.gettempdir()) / "fraud_detection_models"
TEMP_PROCESSED_DIR = Path(tempfile.gettempdir()) / "fraud_detection_processed"
TEMP_ANALYTICS_DIR = Path(tempfile.gettempdir()) / "fraud_detection_analytics"


def existing_file(project_path: Path, temp_dir: Path) -> Path:
    """Return project file when present, otherwise matching temp file."""
    if project_path.exists():
        return project_path
    temp_path = temp_dir / project_path.name
    return temp_path if temp_path.exists() else project_path
