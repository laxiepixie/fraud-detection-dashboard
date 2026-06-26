import os
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

# locate dashboard_predictions.csv (temp processed dir or project processed dir)
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

models = [
    ("Classifier (lgbm)", "pred_lgbm"),
    ("Isolation Forest", "pred_if"),
    ("Local Outlier Factor", "pred_lof"),
    ("Mahalanobis (cpp)", "pred_cpp"),
    ("SVM (Linear)", "pred_svm")
]

for name, col in models:
    print(f"\n{name}:")
    if col not in df.columns:
        print(f"  Prediction column '{col}' not found")
        continue
    y_pred = df[col].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    total = tn+fp+fn+tp
    print(f"  Total={total}  Accuracy={(tn+tp)/max(total,1):.4f}")
