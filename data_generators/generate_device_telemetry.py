"""
generate_device_telemetry.py
Generates synthetic device telemetry time-series data for the TORUS-2.0 project.
Simulates CPU/memory utilization, error counts, and uptime for each ultrasound device.
Saves output to data/raw/device_telemetry.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_device_telemetry(
    devices=None,
    days: int = 90,
    readings_per_day: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    if devices is None:
        devices = ["DEV-001", "DEV-002", "DEV-003", "DEV-004"]

    np.random.seed(seed)

    start_date = datetime(2025, 1, 1)
    records = []

    for device_id in devices:
        # Give each device a baseline "health" profile
        baseline_cpu = np.random.uniform(30, 50)
        baseline_mem = np.random.uniform(40, 60)
        anomaly_device = np.random.rand() < 0.25  # 25% chance this device has anomaly patterns

        for day in range(days):
            for hour_block in range(readings_per_day):
                timestamp = start_date + timedelta(days=day, hours=hour_block)

                cpu_util = baseline_cpu + np.random.normal(0, 5)
                mem_util = baseline_mem + np.random.normal(0, 5)
                error_count = np.random.poisson(0.1)
                uptime_flag = 1

                # Inject anomaly patterns for some devices
                if anomaly_device and np.random.rand() < 0.05:
                    cpu_util += np.random.uniform(30, 50)
                    mem_util += np.random.uniform(20, 40)
                    error_count += np.random.randint(3, 10)
                    uptime_flag = np.random.choice([0, 1], p=[0.3, 0.7])

                records.append({
                    "device_id": device_id,
                    "timestamp": timestamp,
                    "cpu_utilization": round(max(0, min(cpu_util, 100)), 2),
                    "memory_utilization": round(max(0, min(mem_util, 100)), 2),
                    "error_count": int(max(error_count, 0)),
                    "uptime_flag": uptime_flag,
                })

    telemetry_df = pd.DataFrame(records)
    return telemetry_df


if __name__ == "__main__":
    telemetry_df = generate_device_telemetry(days=90, readings_per_day=24)

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/device_telemetry.csv"
    telemetry_df.to_csv(output_path, index=False)

    print(f"Generated {len(telemetry_df)} telemetry records -> {output_path}")
