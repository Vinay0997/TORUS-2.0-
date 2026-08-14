"""
Step 3 - MLflow tracking for device anomaly detection (Isolation Forest).

Place this file at: anomaly_detection/train_anomaly_with_mlflow.py
Run it from repo root:  python anomaly_detection/train_anomaly_with_mlflow.py

Adjust TELEMETRY_FEATURES_PATH and FEATURE_COLS to match whatever your
detect_anomalies.py currently reads (likely device_uptime_stats or similar).
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest

TELEMETRY_FEATURES_PATH = "data/processed/device_uptime_stats.csv"  # adjust if different
FEATURE_COLS = None  # None => use all numeric cols except device_id

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "torus-device-anomaly-detection"


def load_data():
    df = pd.read_csv(TELEMETRY_FEATURES_PATH)
    if FEATURE_COLS is None:
        drop_cols = [c for c in df.columns if c.lower() in ("device_id", "id")]
        feature_cols = [c for c in df.select_dtypes(include=["number"]).columns
                         if c not in drop_cols]
    else:
        feature_cols = FEATURE_COLS
    return df, feature_cols


def run_experiment(n_estimators=100, contamination=0.05, random_state=42):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, feature_cols = load_data()
    X = df[feature_cols]

    with mlflow.start_run(run_name=f"isoforest_n={n_estimators}_cont={contamination}"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_columns", ",".join(feature_cols))

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(X)

        scores = model.decision_function(X)
        preds = model.predict(X)  # -1 = anomaly, 1 = normal

        anomaly_rate = (preds == -1).mean()
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("mean_score", scores.mean())
        mlflow.log_metric("min_score", scores.min())

        results = df.copy()
        results["anomaly_score"] = scores
        results["is_anomaly"] = preds == -1
        results_path = "anomaly_results.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        os.remove(results_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        top_anomalies = results.sort_values("anomaly_score").head(5)
        print(f"Run logged: anomaly_rate={anomaly_rate:.4f}")
        print("Top 5 anomalous devices:\n", top_anomalies)
        return anomaly_rate


if __name__ == "__main__":
    for contamination in [0.02, 0.05, 0.1]:
        run_experiment(contamination=contamination)

    print("\nDone. Launch the MLflow UI with:")
    print("  mlflow ui --backend-store-uri file:./mlruns")
    print("Then open http://127.0.0.1:5000 to compare runs.")
