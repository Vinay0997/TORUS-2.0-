# MLflow Experiment Tracking in TORUS

## Purpose
MLflow provides reproducible tracking of model training runs —
hyperparameters, metrics, and model artifacts — so different model
configurations can be compared objectively rather than relying on
console output that disappears after the script finishes.

## Where it's used

### Follow-up prediction (`models/train_followup_with_mlflow.py`)
Logs two runs under the `torus-followup-prediction` experiment:
- `logistic_regression` — with params max_iter=1000, class_weight=balanced
- `random_forest` — with params n_estimators=200, max_depth=8,
  class_weight=balanced

Each run logs accuracy, precision, recall, F1 score, and ROC-AUC, plus
the fitted sklearn pipeline as a model artifact.

### Device anomaly detection (`anomaly_detection/detect_anomalies_mlflow.py`)
Logs multiple runs under the `torus-device-anomaly-detection` experiment,
sweeping the Isolation Forest `contamination` parameter (0.02, 0.05, 0.1)
to show how sensitivity changes the flagged anomaly rate. Each run logs
overall anomaly rate, mean anomaly score, number of devices, and the
maximum per-device anomaly rate, plus the fitted model and a summary CSV
artifact.

## Local tracking store
Both scripts use a local file-based tracking URI (`file:./mlruns`), so no
external server is required. Runs are viewed via:

```
python -m mlflow ui --port 5001
```

then opening `http://127.0.0.1:5001` in a browser.

## Why compare multiple runs instead of just picking a winner
Logging both candidate models (rather than only the best one) lets a
reviewer see the actual tradeoffs — for example, whether Random Forest's
F1 improvement over Logistic Regression is large enough to justify its
reduced interpretability, or whether a lower Isolation Forest
contamination setting meaningfully changes which devices get flagged.
