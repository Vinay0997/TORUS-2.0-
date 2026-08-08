"""
detect_anomalies.py
TORUS-2.0: Step 3 - Anomaly Detection on Device Telemetry

Reads telemetry data from the warehouse (fact_device_telemetry table),
engineers rolling-window features per device, fits an Isolation Forest
across all devices to flag anomalous readings, and writes results to:
  - data/processed/telemetry_anomalies.csv
  - data/processed/anomaly_summary.csv
  - data/processed/charts/*.png (auto-generated visualizations)

Run this AFTER the Airflow ETL DAG has successfully loaded telemetry data.
"""

import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote_plus

# ---------------------------------------------------------------------
# Config - adjust connection string to match your local Postgres setup
# ---------------------------------------------------------------------
DB_USER = os.getenv("TORUS_DB_USER", "postgres")
DB_PASSWORD = os.getenv("TORUS_DB_PASSWORD", "Yaniv305@#")
DB_HOST = os.getenv("TORUS_DB_HOST", "localhost")
DB_PORT = os.getenv("TORUS_DB_PORT", "5432")
DB_NAME = os.getenv("TORUS_DB_NAME", "TORUS_db")

DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)
CONN_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ROLLING_WINDOW = 6          # number of readings (hours) for rolling stats
CONTAMINATION = 0.05        # expected proportion of anomalies (~5%)


def load_telemetry(engine) -> pd.DataFrame:
    query = """
        SELECT device_id, "timestamp", cpu_utilization,
               memory_utilization, error_count, uptime_flag
        FROM fact_device_telemetry
        ORDER BY device_id, "timestamp"
    """
    df = pd.read_sql(query, engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean/std features per device to capture deviations
    from each device's own recent baseline behavior.
    """
    df = df.sort_values(["device_id", "timestamp"]).copy()

    grouped = df.groupby("device_id")

    df["cpu_roll_mean"] = grouped["cpu_utilization"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["cpu_roll_std"] = grouped["cpu_utilization"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    )
    df["mem_roll_mean"] = grouped["memory_utilization"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["mem_roll_std"] = grouped["memory_utilization"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    )

    df["cpu_deviation"] = df["cpu_utilization"] - df["cpu_roll_mean"]
    df["mem_deviation"] = df["memory_utilization"] - df["mem_roll_mean"]

    df["error_roll_sum"] = grouped["error_count"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).sum()
    )

    return df


def detect_anomalies_per_device(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits ONE Isolation Forest across all devices (using device-relative
    deviation features), then flags anomalies using a fixed score
    threshold rather than a forced per-device contamination rate.
    This lets clean devices legitimately score near-zero anomalies
    while devices with real injected spikes stand out.
    """
    feature_cols = [
        "cpu_utilization", "memory_utilization", "error_count",
        "cpu_deviation", "mem_deviation", "error_roll_sum", "uptime_flag",
    ]

    X = df[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(X)
    scores = model.decision_function(X)  # lower score = more anomalous

    df = df.copy()
    df["anomaly_score"] = scores

    threshold = np.percentile(scores, CONTAMINATION * 100)
    df["anomaly_flag"] = (df["anomaly_score"] < threshold).astype(int)

    return df


def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("device_id")["anomaly_flag"]
        .agg(total_readings="count", anomaly_count="sum")
        .reset_index()
    )
    summary["anomaly_rate_pct"] = round(
        summary["anomaly_count"] / summary["total_readings"] * 100, 2
    )
    return summary.sort_values("anomaly_rate_pct", ascending=False)


def plot_device_timeseries(df: pd.DataFrame, device_id: str, output_dir: str):
    """
    Saves a CPU/memory time-series chart for a single device with
    anomaly-flagged points highlighted in red.
    """
    dev_df = df[df["device_id"] == device_id].copy()
    normal = dev_df[dev_df["anomaly_flag"] == 0]
    anomalous = dev_df[dev_df["anomaly_flag"] == 1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=("CPU Utilization (%)", "Memory Utilization (%)"),
    )

    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["cpu_utilization"],
                              mode="lines", name="Normal", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalous["timestamp"], y=anomalous["cpu_utilization"],
                              mode="markers", name="Anomaly", marker=dict(size=7, color="red")), row=1, col=1)

    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["memory_utilization"],
                              mode="lines", name="Normal", line=dict(width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=anomalous["timestamp"], y=anomalous["memory_utilization"],
                              mode="markers", name="Anomaly", marker=dict(size=7, color="red"), showlegend=False), row=2, col=1)

    fig.update_layout(
        title={"text": f"{device_id} Telemetry Anomalies (90-Day Period)"
                        "<br><span style='font-size: 18px; font-weight: normal;'>"
                        "Source: TORUS-2.0 | Isolation Forest flagged spikes in red</span>"},
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font=dict(size=16)),
        font=dict(size=16),
    )
    fig.update_annotations(font_size=18)
    fig.update_xaxes(title_text="Date", tickfont=dict(size=14), title_font=dict(size=16), row=1, col=1)
    fig.update_xaxes(title_text="Date", tickfont=dict(size=14), title_font=dict(size=16), row=2, col=1)
    fig.update_yaxes(title_text="CPU %", tickfont=dict(size=14), title_font=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text="Memory %", tickfont=dict(size=14), title_font=dict(size=16), row=2, col=1)
    fig.update_traces(cliponaxis=False)

    safe_name = device_id.lower().replace("-", "")
    png_path = os.path.join(output_dir, f"{safe_name}_anomalies_timeseries.png")
    fig.write_image(png_path, width=1400, height=900, scale=2)
    with open(png_path + ".meta.json", "w") as f:
        json.dump({
            "caption": f"{device_id} CPU/memory anomalies over 90 days",
            "description": f"Time series of {device_id} CPU and memory utilization with anomalies highlighted in red",
        }, f)
    print(f"Saved chart -> {png_path}")


def plot_anomaly_rate_by_device(summary: pd.DataFrame, output_dir: str):
    """
    Saves a bar chart comparing anomaly rate percentage across all devices.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["device_id"], y=summary["anomaly_rate_pct"],
        text=summary["anomaly_rate_pct"].astype(str) + "%",
        textposition="outside", textfont=dict(size=16),
    ))
    fig.update_layout(
        title={"text": "Anomaly Rate by Device (90-Day Telemetry)"
                        "<br><span style='font-size: 18px; font-weight: normal;'>"
                        "Source: TORUS-2.0 anomaly detection pipeline</span>"},
        font=dict(size=16),
    )
    max_rate = summary["anomaly_rate_pct"].max()
    fig.update_xaxes(title_text="Device ID", tickfont=dict(size=15), title_font=dict(size=16))
    fig.update_yaxes(title_text="Anomaly Rate (%)", tickfont=dict(size=15),
                      title_font=dict(size=16), range=[0, max_rate * 1.25 if max_rate > 0 else 5])
    fig.update_traces(cliponaxis=False)

    png_path = os.path.join(output_dir, "anomaly_rate_by_device.png")
    fig.write_image(png_path, width=1200, height=800, scale=2)
    with open(png_path + ".meta.json", "w") as f:
        json.dump({
            "caption": "Anomaly rate percentage by device",
            "description": "Bar chart comparing anomaly rates across devices",
        }, f)
    print(f"Saved chart -> {png_path}")


def main():
    engine = create_engine(CONN_STRING)

    print("Loading telemetry data from warehouse...")
    df = load_telemetry(engine)
    print(f"Loaded {len(df)} telemetry rows for {df['device_id'].nunique()} devices")

    print("Engineering rolling-window features...")
    df = engineer_features(df)

    print("Fitting Isolation Forest per device...")
    df = detect_anomalies_per_device(df)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/telemetry_anomalies.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved scored telemetry -> {output_path}")

    summary = summarize_anomalies(df)
    summary_path = "data/processed/anomaly_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved device anomaly summary -> {summary_path}")

    print("\nAnomaly summary by device:")
    print(summary.to_string(index=False))

    charts_dir = "data/processed/charts"
    os.makedirs(charts_dir, exist_ok=True)

    top_device = summary.sort_values("anomaly_rate_pct", ascending=False).iloc[0]["device_id"]
    plot_device_timeseries(df, top_device, charts_dir)
    plot_anomaly_rate_by_device(summary, charts_dir)

    print(f"\nCharts saved to {charts_dir}/")


if __name__ == "__main__":
    main()