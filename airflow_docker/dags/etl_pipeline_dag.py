"""
etl_pipeline_dag.py
TORUS-2.0: Airflow DAG for extract -> transform/validate -> load pipeline.

Orchestrates loading of patients.csv, ultrasound_exams.csv, and
device_telemetry.csv from data/raw/ into the Redshift/Postgres warehouse
tables defined in redshift_ddl.sql.

Place this file in your Airflow dags/ folder.
"""

from datetime import datetime, timedelta
import os
import json

import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
RAW_DATA_DIR = os.getenv("TORUS_RAW_DATA_DIR", "/opt/airflow/data/raw")
DB_CONN_ID = "torus_warehouse"  # Airflow Connection ID configured in UI

default_args = {
    "owner": "torus_team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------
# Extract tasks
# ---------------------------------------------------------------------
def extract_csv(filename: str, **context):
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected raw file not found: {path}")

    df = pd.read_csv(path)
    context["ti"].xcom_push(key=filename, value=df.to_json(orient="records"))
    print(f"Extracted {len(df)} rows from {filename}")


# ---------------------------------------------------------------------
# Transform / validate tasks
# ---------------------------------------------------------------------
def transform_validate_patients(**context):
    raw_json = context["ti"].xcom_pull(key="patients.csv", task_ids="extract_patients")
    df = pd.read_json(raw_json, orient="records")

    df = df.dropna(subset=["patient_id", "age", "sex", "risk_category"])
    df = df[(df["age"] >= 0) & (df["age"] <= 120)]
    df = df.drop_duplicates(subset=["patient_id"])

    assert df["patient_id"].is_unique, "Duplicate patient_id found after dedup"

    context["ti"].xcom_push(key="patients_clean", value=df.to_json(orient="records"))
    print(f"Validated {len(df)} clean patient rows")


def transform_validate_exams(**context):
    raw_json = context["ti"].xcom_pull(key="ultrasound_exams.csv", task_ids="extract_exams")
    df = pd.read_json(raw_json, orient="records")

    df = df.dropna(subset=["exam_id", "patient_id", "device_id", "exam_timestamp"])
    df = df[(df["image_quality_score"] >= 0) & (df["image_quality_score"] <= 1)]
    df = df.drop_duplicates(subset=["exam_id"])

    context["ti"].xcom_push(key="exams_clean", value=df.to_json(orient="records"))
    print(f"Validated {len(df)} clean exam rows")


def transform_validate_telemetry(**context):
    raw_json = context["ti"].xcom_pull(key="device_telemetry.csv", task_ids="extract_telemetry")
    df = pd.read_json(raw_json, orient="records")

    df = df.dropna(subset=["device_id", "timestamp"])
    df["cpu_utilization"] = df["cpu_utilization"].clip(0, 100)
    df["memory_utilization"] = df["memory_utilization"].clip(0, 100)
    df["error_count"] = df["error_count"].clip(lower=0)

    context["ti"].xcom_push(key="telemetry_clean", value=df.to_json(orient="records"))
    print(f"Validated {len(df)} clean telemetry rows")


# ---------------------------------------------------------------------
# Load tasks
# ---------------------------------------------------------------------
def load_table(xcom_key: str, upstream_task_id: str, table_name: str, **context):
    """
    Loads a cleaned dataframe into the warehouse.
    Uses the Airflow Postgres/Redshift hook via DB_CONN_ID.
    """
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    clean_json = context["ti"].xcom_pull(key=xcom_key, task_ids=upstream_task_id)
    df = pd.read_json(clean_json, orient="records")

    hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    engine = hook.get_sqlalchemy_engine()

    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi")
    print(f"Loaded {len(df)} rows into {table_name}")




# ---------------------------------------------------------------------
# Anomaly detection task (Step 3 integration)
# ---------------------------------------------------------------------
def run_anomaly_detection(**context):
    """
    Runs Isolation Forest anomaly detection on freshly-loaded telemetry
    data and saves scored CSVs + charts, same logic as the standalone
    detect_anomalies.py script, so it executes automatically after
    each successful ETL load.
    """
    from sklearn.ensemble import IsolationForest
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    engine = hook.get_sqlalchemy_engine()

    query = """
        SELECT device_id, "timestamp", cpu_utilization,
               memory_utilization, error_count, uptime_flag
        FROM fact_device_telemetry
        ORDER BY device_id, "timestamp"
    """
    df = pd.read_sql(query, engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["device_id", "timestamp"])

    rolling_window = 6
    grouped = df.groupby("device_id")
    df["cpu_roll_mean"] = grouped["cpu_utilization"].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean()
    )
    df["mem_roll_mean"] = grouped["memory_utilization"].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean()
    )
    df["cpu_deviation"] = df["cpu_utilization"] - df["cpu_roll_mean"]
    df["mem_deviation"] = df["memory_utilization"] - df["mem_roll_mean"]
    df["error_roll_sum"] = grouped["error_count"].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).sum()
    )

    feature_cols = [
        "cpu_utilization", "memory_utilization", "error_count",
        "cpu_deviation", "mem_deviation", "error_roll_sum", "uptime_flag",
    ]
    X = df[feature_cols].fillna(0)

    contamination = 0.05
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    df["anomaly_score"] = scores
    threshold = np.percentile(scores, contamination * 100)
    df["anomaly_flag"] = (df["anomaly_score"] < threshold).astype(int)

    output_dir = os.getenv("TORUS_PROCESSED_DIR", "/opt/airflow/data/processed")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "telemetry_anomalies.csv"), index=False)

    summary = (
        df.groupby("device_id")["anomaly_flag"]
        .agg(total_readings="count", anomaly_count="sum")
        .reset_index()
    )
    summary["anomaly_rate_pct"] = round(
        summary["anomaly_count"] / summary["total_readings"] * 100, 2
    )
    summary = summary.sort_values("anomaly_rate_pct", ascending=False)
    summary.to_csv(os.path.join(output_dir, "anomaly_summary.csv"), index=False)

    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    top_device = summary.iloc[0]["device_id"]
    dev_df = df[df["device_id"] == top_device]
    normal = dev_df[dev_df["anomaly_flag"] == 0]
    anomalous = dev_df[dev_df["anomaly_flag"] == 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                         subplot_titles=("CPU Utilization (%)", "Memory Utilization (%)"))
    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["cpu_utilization"],
                              mode="lines", name="Normal", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalous["timestamp"], y=anomalous["cpu_utilization"],
                              mode="markers", name="Anomaly", marker=dict(size=7, color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["memory_utilization"],
                              mode="lines", name="Normal", line=dict(width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=anomalous["timestamp"], y=anomalous["memory_utilization"],
                              mode="markers", name="Anomaly", marker=dict(size=7, color="red"), showlegend=False), row=2, col=1)
    fig.update_layout(
        title={"text": f"{top_device} Telemetry Anomalies (Latest Run)"},
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font=dict(size=16)),
        font=dict(size=16),
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="CPU %", row=1, col=1)
    fig.update_yaxes(title_text="Memory %", row=2, col=1)

    safe_name = top_device.lower().replace("-", "")
    png_path = os.path.join(charts_dir, f"{safe_name}_anomalies_timeseries.png")
    fig.write_image(png_path, width=1400, height=900, scale=2)

    print(f"Anomaly detection complete. Top device: {top_device}")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="torus_etl_pipeline",
    default_args=default_args,
    description="Extract -> transform/validate -> load pipeline for TORUS-2.0",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["torus", "etl", "warehouse"],
) as dag:

    extract_patients = PythonOperator(
        task_id="extract_patients",
        python_callable=extract_csv,
        op_kwargs={"filename": "patients.csv"},
    )

    extract_exams = PythonOperator(
        task_id="extract_exams",
        python_callable=extract_csv,
        op_kwargs={"filename": "ultrasound_exams.csv"},
    )

    extract_telemetry = PythonOperator(
        task_id="extract_telemetry",
        python_callable=extract_csv,
        op_kwargs={"filename": "device_telemetry.csv"},
    )

    validate_patients = PythonOperator(
        task_id="validate_patients",
        python_callable=transform_validate_patients,
    )

    validate_exams = PythonOperator(
        task_id="validate_exams",
        python_callable=transform_validate_exams,
    )

    validate_telemetry = PythonOperator(
        task_id="validate_telemetry",
        python_callable=transform_validate_telemetry,
    )

    load_patients = PythonOperator(
        task_id="load_patients",
        python_callable=load_table,
        op_kwargs={
            "xcom_key": "patients_clean",
            "upstream_task_id": "validate_patients",
            "table_name": "dim_patients",
        },
    )

    load_exams = PythonOperator(
        task_id="load_exams",
        python_callable=load_table,
        op_kwargs={
            "xcom_key": "exams_clean",
            "upstream_task_id": "validate_exams",
            "table_name": "fact_ultrasound_exams",
        },
    )

    load_telemetry = PythonOperator(
        task_id="load_telemetry",
        python_callable=load_table,
        op_kwargs={
            "xcom_key": "telemetry_clean",
            "upstream_task_id": "validate_telemetry",
            "table_name": "fact_device_telemetry",
        },
    )


    detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=run_anomaly_detection,
    )

    # Dependency chain: extract -> validate -> load, per dataset.
    # Patients must load before exams (FK dependency on patient_id).
    extract_patients >> validate_patients >> load_patients
    extract_exams >> validate_exams >> load_exams
    extract_telemetry >> validate_telemetry >> load_telemetry

    load_patients >> load_exams
    load_telemetry >> detect_anomalies
