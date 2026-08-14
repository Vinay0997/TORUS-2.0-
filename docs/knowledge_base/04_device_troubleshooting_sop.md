# Device Troubleshooting SOP

## Purpose
This SOP describes how to interpret anomaly flags raised by the TORUS
anomaly detection pipeline (`anomaly_detection/detect_anomalies.py`) and
what steps a technician should take when a device is flagged.

## Step 1: Confirm the anomaly is real, not a data gap
Check `data/processed/anomaly_summary.csv` for the flagged device's
`anomaly_rate_pct`. A device with very few total readings and one flagged
point may reflect noisy data rather than a real fault. Cross-reference
against `total_readings` in the same row — low readings + high anomaly
rate percentage is a data-quality flag, not necessarily a device fault.

## Step 2: Review the time-series chart
Charts saved to `data/processed/charts/{device_id}_anomalies_timeseries.png`
show CPU and memory utilization over time with anomalous points marked in
red. Look for:
- **Sustained spikes**: CPU/memory staying elevated for multiple
  consecutive readings — suggests a real performance issue (e.g., stuck
  process, memory leak).
- **Isolated single-point spikes**: A single anomalous reading surrounded
  by normal readings — often transient (e.g., a background OS update)
  and lower priority.

## Step 3: Correlate with error_count
High `error_count` alongside CPU/memory deviation is a stronger signal of
a genuine device fault than utilization deviation alone. Devices with
rising `error_roll_sum` (6-hour rolling error sum) should be prioritized
for physical inspection.

## Step 4: Escalation thresholds
- **anomaly_rate_pct > 10%**: Escalate to on-site technician inspection
  within 24 hours.
- **anomaly_rate_pct 5–10%**: Flag for monitoring; re-check in the next
  scheduled anomaly detection run.
- **anomaly_rate_pct < 5%**: Likely within normal variation; no action
  required unless corroborated by error_count spikes.

## Step 5: Document findings
Record the device_id, date range reviewed, and resolution (e.g.,
"rebooted device," "replaced probe cable," "false positive — no action")
in the device maintenance log.

## Known limitation
The current anomaly detector fits a single Isolation Forest across all
devices using device-relative deviation features. It does not yet
distinguish between a device with a genuinely different failure mode and
one experiencing a temporary environmental factor (e.g., ambient
temperature in a mobile clinic van). This is a candidate for a future
per-device model refinement.
