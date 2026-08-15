# TORUS 2.0 — Remote Ultrasound AI Platform

🔗 **[Live Dashboard](https://torus-anomaly-dashboard.streamlit.app)** — interactive anomaly detection, follow-up prediction, and RAG assistant, all in one app.
🔗 **[Live RAG Assistant](https://torus-rag-assistant.streamlit.app/)** — standalone chat assistant for the TORUS architecture, FHIR/HL7 concepts, and the ML models.

TORUS 2.0 is an end-to-end data engineering and machine learning pipeline that simulates a remote ultrasound telehealth platform: synthetic patient, exam, and device telemetry data flowing through an orchestrated ETL pipeline into a data warehouse, with automated anomaly detection on device health metrics, a follow-up prediction model, and a retrieval-augmented generation assistant for exploring the project itself.

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
                              │ Unified Streamlit App    │
                              │ (anomaly + follow-up +   │
                              │  RAG assistant tabs)     │
                              └───────────────────────┘
```

## What it does

1. **Synthetic data generation** — Python scripts generate realistic patient records, ultrasound exam logs, and device telemetry (CPU, memory, error counts, uptime) for a fleet of remote ultrasound devices.
2. **Orchestrated ETL** — An Apache Airflow DAG (`torus_etl_pipeline`) extracts, validates, and loads each dataset into a Postgres/Redshift-style warehouse schema, running on a schedule inside Docker.
3. **Anomaly detection** — After telemetry loads, a `detect_anomalies` task automatically runs an Isolation Forest model over rolling-window CPU/memory/error features to flag anomalous device behavior, without any manual script execution.
4. **Automated reporting** — Each run produces a per-device anomaly summary CSV, a full scored telemetry CSV, and time-series charts highlighting flagged anomalies (e.g. device DEV-002).
5. **Unified interactive dashboard** — A single Streamlit app with three tabs: fleet-wide anomaly browsing, follow-up prediction results, and a RAG assistant chat — all live in the browser.
6. **Follow-up prediction** — A trained model predicts patient follow-up outcomes, with experiments tracked via MLflow.
7. **RAG assistant** — A chatbot answers questions about the project's own architecture and documentation, grounded strictly in a FAISS-indexed knowledge base.

## Project structure

| Folder | Contents |
|---|---|
| `data_generators/` | Scripts generating synthetic patients, exams, and device telemetry |
| `airflow_docker/` | Docker Compose setup, DAG definitions, and Airflow configuration |
| `anomaly_detection/` | Standalone Isolation Forest anomaly detection script |
| `dashboard/` | Unified Streamlit app (`app.py`) — anomaly detection, follow-up prediction, and RAG assistant tabs |
| `rag_assistant/` | RAG chatbot: FAISS index builder, LangChain RAG chain, and standalone Streamlit chat UI |
| `models/` | Trained model artifacts |
| `warehouse/` | Redshift/Postgres DDL for the warehouse schema |
| `schemas/`, `sql/` | Supporting schema and SQL definitions |
| `data/raw/`, `data/processed/` | Generated raw data and anomaly detection outputs/charts |
| `docs/` | Project documentation, source material for the RAG assistant's knowledge base |
| `04_build_modeling_dataset.ipynb`, `05_train_followup_prediction.ipynb` | Modeling notebooks for the follow-up prediction task |

## Tech stack

Python, Apache Airflow 3, Docker & Docker Compose, PostgreSQL/Redshift, scikit-learn (Isolation Forest), Streamlit, Plotly, Pandas, NumPy, LangChain, Hugging Face Inference Providers, FAISS, sentence-transformers, MLflow.

## Unified Dashboard

The live unified app is deployed on Streamlit Community Cloud: **[torus-anomaly-dashboard.streamlit.app](https://torus-anomaly-dashboard.streamlit.app)**

It combines three tabs in a single app:
- **📡 Anomaly Detection** — fleet-wide anomaly rates and per-device telemetry drill-down, reading from `data/processed/telemetry_anomalies.csv` and `anomaly_summary.csv`.
- **📈 Follow-up Prediction** — model comparison results and optional live inference if a saved model artifact exists at `models/followup_model.pkl`.
- **💬 RAG Assistant** — the same grounded chatbot described below, reusing `rag_assistant/ask_question.py`'s chain directly.

To run it locally:

```bash
pip install -r dashboard/requirements.txt
python rag_assistant/build_vector_store.py   # builds the FAISS index used by the RAG tab
$env:HUGGINGFACEHUB_API_TOKEN = "your_token_here"
streamlit run dashboard/app.py
```

## RAG Assistant

The RAG assistant is also available as a standalone app: **[torus-rag-assistant.streamlit.app](https://torus-rag-assistant.streamlit.app/)**

A retrieval-augmented generation chatbot built on top of the project's own documentation and codebase. It answers questions strictly from a FAISS-indexed knowledge base, and explicitly declines to guess or provide clinical advice when a question falls outside scope — since all data in this project is synthetic and non-PHI.

**How it works:**
1. Project docs/code are chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2`.
2. Embeddings are stored in a local FAISS vector index (`rag_assistant/faiss_index/`).
3. On each question, the top-k relevant chunks are retrieved and passed as context to `Qwen/Qwen2.5-7B-Instruct`, served via Hugging Face's Inference Providers, through a LangChain RAG chain.
4. A system prompt enforces grounded, non-clinical answers only.

FAISS was chosen over a hosted vector database (e.g. pgvector) intentionally: the knowledge base is small and static, and a local index avoids exposing database credentials from a public Streamlit app while keeping the deployment self-contained.

**Run it standalone:**
```bash
pip install -r rag_assistant/requirements.txt
python rag_assistant/build_vector_store.py   # builds the FAISS index
$env:HUGGINGFACEHUB_API_TOKEN = "your_token_here"
streamlit run rag_assistant/streamlit_app.py
```

## Follow-up Prediction Model

Two models are trained and compared for predicting patient follow-up outcomes (`04_build_modeling_dataset.ipynb`, `05_train_followup_prediction.ipynb`), with experiment runs tracked via **MLflow** — logging metrics and artifacts across training runs to select the best-performing model.

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

- [x] Integrate the RAG assistant, anomaly dashboard, and follow-up prediction results into a single unified Streamlit app
- [ ] Alerting layer (Slack/email) when a device's anomaly rate crosses a threshold
- [ ] GitHub Actions CI to lint/test the DAG and data generators on every push
- [ ] Export a saved follow-up prediction model artifact (`models/followup_model.pkl`) and metrics summary (`models/followup_metrics.json`) to enable live inference in the dashboard's Follow-up Prediction tab
