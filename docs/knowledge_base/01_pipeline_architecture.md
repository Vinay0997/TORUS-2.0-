# TORUS 2.0 Pipeline Architecture

## Overview
TORUS 2.0 is a synthetic-data platform simulating remote ultrasound device
monitoring and patient exam outcome prediction. It mirrors a real-world
health-tech data pipeline: raw data generation, warehouse ETL, feature
engineering, ML modeling, and anomaly detection.

## Data flow
1. **Synthetic data generation** (`data_generators/`): Python scripts
   produce three core datasets:
   - `patients` — patient_id, age, sex, risk_category
   - `ultrasound_exams` — exam_id, patient_id, device_id, exam_type,
     timestamp, quality_score, outcome_label (normal / follow-up)
   - `device_telemetry` — device_id, timestamp, cpu_utilization,
     memory_utilization, error_count, uptime_flag

2. **Warehouse layer** (`sql/`, `warehouse/`): Raw CSVs are loaded into a
   PostgreSQL warehouse (`TORUS_db`). Schema DDL lives in
   `sql/01_schema_and_raw_tables.sql`. Feature tables such as
   `fact_device_telemetry` are queried directly by downstream scripts.

3. **Orchestration** (`airflow_docker/`): An Airflow DAG (Docker Compose
   setup) schedules the ETL steps — loading raw CSVs into the warehouse
   and triggering feature-building jobs on a schedule.

4. **Anomaly detection** (`anomaly_detection/detect_anomalies.py`): Reads
   telemetry from `fact_device_telemetry`, engineers rolling-window
   features (6-hour rolling mean/std for CPU and memory, deviation from
   baseline, rolling error sums), and fits an Isolation Forest
   (contamination=0.05, n_estimators=300) across all devices. Anomalies
   are flagged using a percentile-based threshold on the anomaly score
   rather than a forced per-device rate. Results are written to
   `data/processed/telemetry_anomalies.csv` and
   `data/processed/anomaly_summary.csv`, plus visual charts.

5. **Follow-up prediction** (`05_train_followup_prediction.ipynb`,
   `models/`): Reads `data/processed/modeling_dataset.csv` and trains two
   candidate models — a Logistic Regression pipeline and a Random Forest
   pipeline, both using a `ColumnTransformer` with `OneHotEncoder` for
   categorical features. Models are compared on F1 score, and the winner
   is saved to `models/followup_prediction_{best_model_name}.pkl`.

6. **Experiment tracking** (MLflow): Both the follow-up classifier and the
   anomaly detector log hyperparameters, metrics (accuracy, precision,
   recall, F1, ROC-AUC for the classifier; anomaly rate and score
   statistics for the detector), and model artifacts to a local MLflow
   tracking store (`./mlruns`), viewable via `mlflow ui`.

7. **Dashboard** (`dashboard/app.py`): A Streamlit app deployed publicly
   that visualizes anomaly detection results and device health, live at
   the project's Streamlit Community Cloud URL.

## Design intent
This project intentionally mirrors production healthcare data-engineering
patterns (synthetic PHI-free data, warehouse-based ETL, orchestrated
pipelines, tracked ML experiments) without touching real patient data —
useful for demonstrating pipeline design skills in a portfolio context.
