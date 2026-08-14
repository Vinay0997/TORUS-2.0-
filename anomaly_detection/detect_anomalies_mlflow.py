"""
detect_anomalies_mlflow.py
TORUS-2.0: Step 3 (finish) - MLflow tracking wrapper around detect_anomalies.py

Place this file at: anomaly_detection/detect_anomalies_mlflow.py
Run it from repo root:  python anomaly_detection/detect_anomalies_mlflow.py

Reuses load_telemetry() and engineer_features() from your existing
detect_anomalies.py so there is no duplicated logic - this file only
adds MLflow experiment tracking around a parametrized Isolation Forest
sweep (varying contamination) and logs metrics/artifacts/model per run.
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from urllib.parse import quote_plus

from detect_anomalies import load_telemetry, engineer_features, summarize_anomalies

# ---------------------------------------------------------------------
# Config - reads from environment variables (do NOT hardcode secrets)
# ---------------------------------------------------------------------
DB_USER = os.getenv("TORUS_DB_USER", "postgres")
DB_PASSWORD = os.getenv("TORUS_DB_PASSWORD")
DB_HOST = os.getenv("TORUS_DB_HOST", "localhost")
DB_PORT = os.getenv("TORUS_DB_PORT", "5432")
DB_NAME = os.getenv("TORUS_DB_NAME", "TORUS_db")

if not DB_PASSWORD:
    raise RuntimeError(
        "TORUS_DB_PASSWORD environment variable is not set. "
        "Set it before running, e.g.:\n"
        "  $env:TORUS_DB_PASSWORD = \"your_password_here\"   (PowerShell)"
    )

DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)
CONN_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ROLLING_WINDOW = 6
FEATURE_COLS = [
    "cpu_utilization", "memory_utilization", "error_count",
    "cpu_deviation", "mem_deviation", "error_roll_sum", "uptime_flag",
]

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "torus-device-anomaly-detection"


def detect_with_params(df: pd.DataFrame, contamination: float, n_estimators: int):
    X = df[FEATURE_COLS].fillna(0)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)
    scores = model.decision_function(X)

    result = df.copy()
    result["anomaly_score"] = scores
    threshold = np.percentile(scores, contamination * 100)
    result["anomaly_flag"] = (result["anomaly_score"] < threshold).astype(int)
    return result, model


def run_experiment(df_features: pd.DataFrame, contamination: float, n_estimators: int = 300):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"isoforest_cont={contamination}_n={n_estimators}"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("rolling_window", ROLLING_WINDOW)
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLS))

        scored_df, model = detect_with_params(df_features, contamination, n_estimators)
        summary = summarize_anomalies(scored_df)

        overall_anomaly_rate = scored_df["anomaly_flag"].mean()
        mlflow.log_metric("overall_anomaly_rate", overall_anomaly_rate)
        mlflow.log_metric("mean_anomaly_score", scored_df["anomaly_score"].mean())
        mlflow.log_metric("n_devices", scored_df["device_id"].nunique())
        mlflow.log_metric("max_device_anomaly_rate_pct", summary["anomaly_rate_pct"].max())

        summary_path = "mlflow_anomaly_summary_tmp.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="summary")
        os.remove(summary_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"contamination={contamination} n_estimators={n_estimators} "
              f"-> overall_anomaly_rate={overall_anomaly_rate:.4f}")
        print(summary.to_string(index=False))
        return summary


def main():
    engine = create_engine(CONN_STRING)

    print("Loading telemetry data from warehouse...")
    df = load_telemetry(engine)
    print(f"Loaded {len(df)} rows for {df['device_id'].nunique()} devices")

    print("Engineering rolling-window features...")
    df_features = engineer_features(df)

    print("\nRunning MLflow-tracked Isolation Forest sweep...\n")
    for contamination in [0.02, 0.05, 0.1]:
        run_experiment(df_features, contamination=contamination)

    print("\nDone. Launch the MLflow UI with:")
    print("  mlflow ui --backend-store-uri file:./mlruns")
    print("Then open http://127.0.0.1:5000 to compare runs.")


if __name__ == "__main__":
    main()
