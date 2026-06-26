import os
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Try temp processed dir first, then project processed dir
proc_dir = os.path.join(tempfile.gettempdir(), "fraud_detection_processed")
csv_path = os.path.join(proc_dir, "dashboard_predictions.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.abspath(os.path.join(Path(__file__).resolve().parents[2], "data", "processed", "dashboard_predictions.csv"))

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"dashboard_predictions.csv not found (tried {csv_path})")

print(f"Loading predictions from: {csv_path}")
df = pd.read_csv(csv_path)
if 'is_fraud' not in df.columns:
    raise ValueError('is_fraud column not found in predictions file')

y = df['is_fraud'].astype(int)

def report(name, score_col, pred_col):
    if pred_col not in df.columns:
        print(f"{name}: prediction column {pred_col} not found")
        return
    y_pred = df[pred_col].astype(int)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = None
    if score_col in df.columns:
        try:
            auc = roc_auc_score(y, df[score_col])
        except Exception:
            auc = None
    if auc is not None:
        print(f"{name}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} roc_auc={auc:.4f}")
    else:
        print(f"{name}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

report("Classifier (lgbm)", "score_lgbm", "pred_lgbm")
report("Isolation Forest", "score_if", "pred_if")
report("Local Outlier Factor", "score_lof", "pred_lof")
report("Mahalanobis (cpp)", "score_cpp", "pred_cpp")
report("SVM (Linear)", "score_svm", "pred_svm")
