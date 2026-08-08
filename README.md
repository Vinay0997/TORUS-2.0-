# TORUS 2.0 — Remote Ultrasound AI Platform

🔗 **[Live Dashboard](https://torus-anomaly-dashboard.streamlit.app)** — interactive anomaly detection results, no setup required.

TORUS 2.0 is an end-to-end data engineering and machine learning pipeline that simulates a remote ultrasound telehealth platform: synthetic patient, exam, and device telemetry data flowing through an orchestrated ETL pipeline into a data warehouse, with automated anomaly detection on device health metrics.

## Architecture

```
┌─────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│  Data Generators     │────▶│  Airflow ETL DAG    │────▶│  Postgres/Redshift   │
│  patients, exams,     │     │  extract→validate→  │     │  Warehouse           │
│  device telemetry     │     │  load (per dataset)  │     │  (fact/dim tables)    │
└─────────────────────┘     └──────────┬──────────┘     └──────────┬───────────┘
                                        │                            │
                                        ▼                            ▼
                              ┌───────────────────┐        ┌──────────────────────┐
                              │ detect_anomalies   │◀───────│  fact_device_telemetry│
                              │ (Isolation Forest)  │        └──────────────────────┘
                              └──────────┬──────────┘
                                         ▼
                              ┌───────────────────────┐
                              │ CSV outputs + charts    │
                              │ (anomaly_summary.csv,   │
                              │  telemetry_anomalies.csv│
                              │  device timeseries PNGs) │
                              └───────────────────────┘
                                         │
                                         ▼
                              ┌───────────────────┐
                              │ Streamlit Dashboard      │
                              │ (live, interactive)      │
                              └───────────────────────┘
```

## What it does

1. **Synthetic data generation** — Python scripts generate realistic patient records, ultrasound exam logs, and device telemetry (CPU, memory, error counts, uptime) for a fleet of remote ultrasound devices.
2. **Orchestrated ETL** — An Apache Airflow DAG (`torus_etl_pipeline`) extracts, validates, and loads each dataset into a Postgres/Redshift-style warehouse schema, running on a schedule inside Docker.
3. **Anomaly detection** — After telemetry loads, a `detect_anomalies` task automatically runs an Isolation Forest model over rolling-window CPU/memory/error features to flag anomalous device behavior, without any manual script execution.
4. **Automated reporting** — Each run produces a per-device anomaly summary CSV, a full scored telemetry CSV, and time-series charts highlighting flagged anomalies (e.g. device DEV-002).
5. **Interactive dashboard** — A Streamlit app lets anyone browse fleet-wide anomaly rates and drill into per-device telemetry time series, live in the browser.

## Project structure

| Folder | Contents |
|---|---|
| `data_generators/` | Scripts generating synthetic patients, exams, and device telemetry |
| `airflow_docker/` | Docker Compose setup, DAG definitions, and Airflow configuration |
| `anomaly_detection/` | Standalone Isolation Forest anomaly detection script |
| `dashboard/` | Streamlit dashboard app (`app.py`) for interactive anomaly browsing |
| `warehouse/` | Redshift/Postgres DDL for the warehouse schema |
| `schemas/`, `sql/` | Supporting schema and SQL definitions |
| `data/raw/`, `data/processed/` | Generated raw data and anomaly detection outputs/charts |
| `04_build_modeling_dataset.ipynb`, `05_train_followup_prediction.ipynb` | Modeling notebooks for downstream ML tasks |

## Tech stack

Python, Apache Airflow 3, Docker & Docker Compose, PostgreSQL/Redshift, scikit-learn (Isolation Forest), Streamlit, Plotly, Pandas, NumPy.

## Dashboard

The live dashboard is deployed on Streamlit Community Cloud: **[torus-anomaly-dashboard.streamlit.app](https://torus-anomaly-dashboard.streamlit.app)**

To run it locally instead:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

It reads directly from `data/processed/telemetry_anomalies.csv` and `anomaly_summary.csv`, letting you filter by device and date range, and toggle an "anomalies only" view over live CPU/memory time-series charts.

## Setup (full pipeline)

### 1. Clone and configure

```bash
git clone https://github.com/Vinay0997/TORUS-2.0-.git
cd TORUS-2.0-/airflow_docker
cp .env.example .env   # fill in real values for DB credentials and FERNET_KEY
```

### 2. Build and start the stack

```bash
docker compose build
docker compose up airflow-init
docker compose up -d
```

### 3. Access Airflow

Open [http://localhost:8080](http://localhost:8080) and trigger the `torus_etl_pipeline` DAG. Tasks run in order: `extract_* → validate_* → load_*` per dataset, with `detect_anomalies` firing automatically after `load_telemetry` completes.

### 4. Review outputs

Check `data/processed/` for:
- `telemetry_anomalies.csv` — full scored telemetry with anomaly flags
- `anomaly_summary.csv` — anomaly rate per device
- `charts/` — time-series visualizations of flagged anomalies

## Results

The Isolation Forest model correctly and consistently flags **DEV-002** as the highest-anomaly-rate device across runs, based on elevated CPU/memory deviation and error-count spikes relative to its rolling baseline — validating the detection logic against known synthetic fault injection.

## Roadmap

- [ ] Alerting layer (Slack/email) when a device's anomaly rate crosses a threshold
- [ ] GitHub Actions CI to lint/test the DAG and data generators on every push
