"""
generate_ultrasound_exams.py
Generates synthetic ultrasound exam metadata for the TORUS-2.0 project.
Depends on data/raw/patients.csv existing (run generate_patients.py first).
Saves output to data/raw/ultrasound_exams.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_exams(patients_df: pd.DataFrame, n_exams: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    devices = ["DEV-001", "DEV-002", "DEV-003", "DEV-004"]
    exam_types = ["abdominal", "cardiac", "obstetric", "vascular"]

    patient_ids_for_exams = np.random.choice(patients_df["patient_id"], n_exams)

    start_date = datetime(2025, 1, 1)
    exam_timestamps = [
        start_date + timedelta(days=int(x), hours=int(np.random.randint(0, 24)))
        for x in np.random.randint(0, 500, n_exams)
    ]

    # Merge risk_category back in in order to bias outcome probabilities realistically
    risk_lookup = patients_df.set_index("patient_id")["risk_category"].to_dict()
    age_lookup = patients_df.set_index("patient_id")["age"].to_dict()

    outcomes = []
    for pid in patient_ids_for_exams:
        risk = risk_lookup.get(pid, "low")
        age = age_lookup.get(pid, 40)

        # Higher risk category and older age -> higher chance of follow_up_required
        followup_prob = 0.15
        if risk == "medium":
            followup_prob = 0.30
        elif risk == "high":
            followup_prob = 0.55
        if age > 60:
            followup_prob += 0.10

        followup_prob = min(followup_prob, 0.85)
        remaining = 1 - followup_prob
        outcome = np.random.choice(
            ["follow_up_required", "normal", "inconclusive"],
            p=[followup_prob, remaining * 0.8, remaining * 0.2]
        )
        outcomes.append(outcome)

    exams_df = pd.DataFrame({
        "exam_id": range(1, n_exams + 1),
        "patient_id": patient_ids_for_exams,
        "device_id": np.random.choice(devices, n_exams),
        "exam_type": np.random.choice(exam_types, n_exams),
        "exam_timestamp": exam_timestamps,
        "image_quality_score": np.round(np.random.uniform(0.5, 1.0, n_exams), 2),
        "outcome_label": outcomes,
    })
    return exams_df


if __name__ == "__main__":
    patients_path = "data/raw/patients.csv"
    if not os.path.exists(patients_path):
        raise FileNotFoundError(
            f"'{patients_path}' not found. Run generate_patients.py first."
        )

    patients_df = pd.read_csv(patients_path)
    exams_df = generate_exams(patients_df, n_exams=800)

    output_path = "data/raw/ultrasound_exams.csv"
    exams_df.to_csv(output_path, index=False)

    print(f"Generated {len(exams_df)} exam records -> {output_path}")
