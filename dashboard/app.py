"""
TORUS 2.0 — Unified Platform Dashboard

Combines three previously separate apps into one multi-tab Streamlit
experience:
  1. Anomaly Detection Dashboard (device telemetry, Isolation Forest)
  2. Follow-up Prediction (model comparison results / live inference)
  3. RAG Assistant (chat with the project's own knowledge base)

Place this file at: dashboard/app.py
Run locally:        streamlit run dashboard/app.py

Requires:
  - data/processed/telemetry_anomalies.csv and anomaly_summary.csv
    (from the anomaly detection pipeline)
  - rag_assistant/faiss_index/ built (run rag_assistant/build_vector_store.py)
  - HUGGINGFACEHUB_API_TOKEN set (env var locally, or st.secrets on Cloud)
  - Optional: models/followup_model.pkl + models/followup_metrics.json
    for live follow-up prediction inference. If absent, the tab falls
    back to a static results summary instead of erroring out.
"""

import os
import sys
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="TORUS 2.0 — Platform Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("TORUS_PROCESSED_DIR", os.path.join(REPO_ROOT, "data", "processed"))
TELEMETRY_PATH = os.path.join(DATA_DIR, "telemetry_anomalies.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "anomaly_summary.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
FOLLOWUP_MODEL_PATH = os.path.join(MODELS_DIR, "followup_model.pkl")
FOLLOWUP_METRICS_PATH = os.path.join(MODELS_DIR, "followup_metrics.json")

st.title("🩺 TORUS 2.0 — Platform Dashboard")
st.caption(
    "Remote ultrasound telehealth platform: device anomaly detection, "
    "follow-up prediction, and a RAG assistant for exploring the project."
)

tab_anomaly, tab_followup, tab_rag = st.tabs(
    ["📡 Anomaly Detection", "📈 Follow-up Prediction", "💬 RAG Assistant"]
)

# =======================================================================
# TAB 1: Anomaly Detection Dashboard
# =======================================================================
with tab_anomaly:

    @st.cache_data(ttl=60)
    def load_anomaly_data():
        if not os.path.exists(TELEMETRY_PATH) or not os.path.exists(SUMMARY_PATH):
            return None, None
        telemetry = pd.read_csv(TELEMETRY_PATH)
        telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
        summary = pd.read_csv(SUMMARY_PATH).sort_values(
            "anomaly_rate_pct", ascending=False
        )
        return telemetry, summary

    telemetry_df, summary_df = load_anomaly_data()

    if telemetry_df is None:
        st.warning(
            "No processed data found yet. Run the ETL pipeline / "
            "`anomaly_detection/detect_anomalies.py` first so that "
            f"`{TELEMETRY_PATH}` and `{SUMMARY_PATH}` exist."
        )
    else:
        st.sidebar.header("Anomaly Detection Filters")

        device_options = summary_df["device_id"].tolist()
        default_device = device_options[0] if device_options else None
        selected_device = st.sidebar.selectbox(
            "Select device", device_options, index=0 if default_device else None
        )

        min_date = telemetry_df["timestamp"].min().date()
        max_date = telemetry_df["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )

        show_only_anomalies = st.sidebar.checkbox("Show only anomalous readings", value=False)

        st.sidebar.divider()
        st.sidebar.metric("Total devices", len(device_options))
        st.sidebar.metric(
            "Overall anomaly rate",
            f"{summary_df['anomaly_count'].sum() / summary_df['total_readings'].sum() * 100:.2f}%",
        )

        st.subheader("Fleet anomaly summary")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_bar = go.Figure(
                go.Bar(
                    x=summary_df["device_id"],
                    y=summary_df["anomaly_rate_pct"],
                    marker_color=[
                        "crimson" if d == selected_device else "steelblue"
                        for d in summary_df["device_id"]
                    ],
                    text=summary_df["anomaly_rate_pct"].astype(str) + "%",
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                title="Anomaly rate by device",
                xaxis_title="Device",
                yaxis_title="Anomaly rate (%)",
                height=420,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.dataframe(
                summary_df.rename(
                    columns={
                        "device_id": "Device",
                        "total_readings": "Readings",
                        "anomaly_count": "Anomalies",
                        "anomaly_rate_pct": "Rate (%)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

        st.divider()

        st.subheader(f"Telemetry detail — {selected_device}")

        dev_df = telemetry_df[telemetry_df["device_id"] == selected_device].copy()

        if len(date_range) == 2:
            start_date, end_date = date_range
            dev_df = dev_df[
                (dev_df["timestamp"].dt.date >= start_date)
                & (dev_df["timestamp"].dt.date <= end_date)
            ]

        if show_only_anomalies:
            dev_df = dev_df[dev_df["anomaly_flag"] == 1]

        normal = dev_df[dev_df["anomaly_flag"] == 0]
        anomalous = dev_df[dev_df["anomaly_flag"] == 1]

        m1, m2, m3 = st.columns(3)
        m1.metric("Readings in view", len(dev_df))
        m2.metric("Anomalies in view", len(anomalous))
        m3.metric(
            "Anomaly rate in view",
            f"{(len(anomalous) / len(dev_df) * 100) if len(dev_df) else 0:.2f}%",
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("CPU Utilization (%)", "Memory Utilization (%)"),
        )

        fig.add_trace(
            go.Scatter(
                x=normal["timestamp"], y=normal["cpu_utilization"],
                mode="lines", name="Normal", line=dict(width=1, color="steelblue"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=anomalous["timestamp"], y=anomalous["cpu_utilization"],
                mode="markers", name="Anomaly", marker=dict(size=8, color="red"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=normal["timestamp"], y=normal["memory_utilization"],
                mode="lines", name="Normal", line=dict(width=1, color="steelblue"),
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=anomalous["timestamp"], y=anomalous["memory_utilization"],
                mode="markers", name="Anomaly", marker=dict(size=8, color="red"),
                showlegend=False,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        )
        fig.update_yaxes(title_text="CPU %", row=1, col=1)
        fig.update_yaxes(title_text="Memory %", row=2, col=1)
        fig.update_xaxes(title_text="Timestamp", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View raw telemetry rows"):
            st.dataframe(
                dev_df[
                    ["timestamp", "cpu_utilization", "memory_utilization",
                     "error_count", "anomaly_score", "anomaly_flag"]
                ].sort_values("timestamp", ascending=False),
                hide_index=True,
                use_container_width=True,
            )

# =======================================================================
# TAB 2: Follow-up Prediction
# =======================================================================
with tab_followup:
    st.subheader("Follow-up Prediction Model")
    st.markdown(
        "Two models are trained and compared to predict patient follow-up "
        "outcomes, with experiments tracked via **MLflow** across training runs."
    )

    if os.path.exists(FOLLOWUP_METRICS_PATH):
        with open(FOLLOWUP_METRICS_PATH) as f:
            metrics = json.load(f)
        st.markdown("### Model comparison results")
        st.json(metrics)
    else:
        st.info(
            "No saved metrics file found at `models/followup_metrics.json`. "
            "Model comparison currently lives in "
            "`05_train_followup_prediction.ipynb` and its MLflow run history.\n\n"
            "To surface live results here, export a summary after training, e.g.:\n\n"
            "```python\n"
            "import json\n"
            "with open('models/followup_metrics.json', 'w') as f:\n"
            "    json.dump({'model_a': {...}, 'model_b': {...}, 'winner': 'model_a'}, f)\n"
            "```"
        )

    st.divider()
    st.markdown("### Live inference")

    if os.path.exists(FOLLOWUP_MODEL_PATH):
        import joblib
        model = joblib.load(FOLLOWUP_MODEL_PATH)
        st.success("Loaded saved follow-up prediction model.")
        st.caption(
            "Upload a CSV of patient/exam records matching the model's "
            "training features to get live predictions."
        )
        uploaded = st.file_uploader("Upload records (CSV)", type="csv")
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            try:
                preds = model.predict(input_df)
                input_df["predicted_followup"] = preds
                st.dataframe(input_df, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed — check that columns match training features.\n\n{e}")
    else:
        st.info(
            "No saved model artifact found at `models/followup_model.pkl`. "
            "Live inference isn't available until the trained model is "
            "exported from the notebook, e.g. `joblib.dump(model, 'models/followup_model.pkl')`."
        )

# =======================================================================
# TAB 3: RAG Assistant
# =======================================================================
with tab_rag:
    st.subheader("💬 TORUS 2.0 Assistant")
    st.caption(
        "Ask about the pipeline architecture, FHIR/HL7 concepts, device "
        "troubleshooting, or the ML models. Answers are grounded in the "
        "project's own knowledge base only."
    )

    rag_dir = os.path.join(REPO_ROOT, "rag_assistant")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)

    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    except Exception:
        pass

    try:
        from ask_question import build_chain, HF_MODEL_REPO_ID
        rag_import_error = None
    except Exception as e:
        rag_import_error = str(e)

    if rag_import_error:
        st.error(
            "Could not load the RAG assistant module. Make sure "
            "`rag_assistant/ask_question.py` and `rag_assistant/faiss_index/` "
            f"exist.\n\nDetail: {rag_import_error}"
        )
    else:
        st.caption(f"Model: `{HF_MODEL_REPO_ID}` via Hugging Face Inference Providers")

        if "rag_chain" not in st.session_state:
            with st.spinner("Loading vector store and connecting to Hugging Face..."):
                try:
                    st.session_state.rag_chain, st.session_state.rag_retriever = build_chain()
                    st.session_state.rag_load_error = None
                except Exception as e:
                    st.session_state.rag_chain, st.session_state.rag_retriever = None, None
                    st.session_state.rag_load_error = str(e)

        if st.session_state.get("rag_load_error"):
            st.error(f"Failed to load the RAG chain: {st.session_state.rag_load_error}")
        else:
            if "rag_messages" not in st.session_state:
                st.session_state.rag_messages = []

            for msg in st.session_state.rag_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("📄 Sources retrieved"):
                            for s in msg["sources"]:
                                st.markdown(f"- `{s}`")

            rag_query = st.chat_input(
                "Ask about TORUS architecture, FHIR/HL7, troubleshooting, or the models...",
                key="rag_chat_input",
            )

            if rag_query:
                st.session_state.rag_messages.append({"role": "user", "content": rag_query})
                with st.chat_message("user"):
                    st.markdown(rag_query)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving context and generating answer..."):
                        try:
                            retrieved_docs = st.session_state.rag_retriever.invoke(rag_query)
                            sources = sorted(set(
                                d.metadata.get("source", "unknown") for d in retrieved_docs
                            ))
                            answer = st.session_state.rag_chain.invoke(rag_query)
                            st.markdown(answer)
                            with st.expander("📄 Sources retrieved"):
                                for s in sources:
                                    st.markdown(f"- `{s}`")
                        except Exception as e:
                            answer = f"Error generating answer: {e}\n\nThis may be a temporary Hugging Face rate limit - try again in ~20 seconds."
                            sources = []
                            st.error(answer)

                st.session_state.rag_messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })

            if st.session_state.rag_messages:
                if st.button("🗑️ Clear RAG conversation"):
                    st.session_state.rag_messages = []
                    st.rerun()
