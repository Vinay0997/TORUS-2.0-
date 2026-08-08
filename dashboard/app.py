"""
TORUS 2.0 — Anomaly Detection Dashboard
Interactive Streamlit app for browsing device telemetry anomaly
detection results produced by anomaly_detection/detect_anomalies.py
(or the equivalent Airflow `detect_anomalies` task).

Run locally:
    streamlit run dashboard/app.py

Expects the following files to exist (produced by the pipeline):
    data/processed/telemetry_anomalies.csv
    data/processed/anomaly_summary.csv
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="TORUS 2.0 — Device Anomaly Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.getenv("TORUS_PROCESSED_DIR", "data/processed")
TELEMETRY_PATH = os.path.join(DATA_DIR, "telemetry_anomalies.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "anomaly_summary.csv")


@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(TELEMETRY_PATH) or not os.path.exists(SUMMARY_PATH):
        return None, None
    telemetry = pd.read_csv(TELEMETRY_PATH)
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
    summary = pd.read_csv(SUMMARY_PATH).sort_values(
        "anomaly_rate_pct", ascending=False
    )
    return telemetry, summary


telemetry_df, summary_df = load_data()

st.title("TORUS 2.0 — Device Anomaly Dashboard")
st.caption(
    "Isolation Forest anomaly detection over remote ultrasound device telemetry."
)

if telemetry_df is None:
    st.warning(
        "No processed data found yet. Run the ETL pipeline / "
        "`detect_anomalies.py` first so that "
        f"`{TELEMETRY_PATH}` and `{SUMMARY_PATH}` exist."
    )
    st.stop()

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Filters")

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

# ---------------------------------------------------------------------
# Fleet-wide summary
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Device drill-down
# ---------------------------------------------------------------------
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
        x=normal["timestamp"],
        y=normal["cpu_utilization"],
        mode="lines",
        name="Normal",
        line=dict(width=1, color="steelblue"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=anomalous["timestamp"],
        y=anomalous["cpu_utilization"],
        mode="markers",
        name="Anomaly",
        marker=dict(size=8, color="red"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=normal["timestamp"],
        y=normal["memory_utilization"],
        mode="lines",
        name="Normal",
        line=dict(width=1, color="steelblue"),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=anomalous["timestamp"],
        y=anomalous["memory_utilization"],
        mode="markers",
        name="Anomaly",
        marker=dict(size=8, color="red"),
        showlegend=False,
    ),
    row=2,
    col=1,
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
            [
                "timestamp",
                "cpu_utilization",
                "memory_utilization",
                "error_count",
                "anomaly_score",
                "anomaly_flag",
            ]
        ].sort_values("timestamp", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
