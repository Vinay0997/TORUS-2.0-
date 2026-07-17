# data_generators/generate_ultrasound_exams.py
import pandas as pd
import numpy as np

def generate_exams(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    patient_df = pd.read_csv("data/raw/patients.csv")
    patient_ids = patient_df["patient_id"].values
    risk_by_patient = dict(zip(patient_df["patient_id"], patient_df["risk_category"]))

    exam_ids = np.arange(1, n+1)
    device_ids = rng.integers(1, 21, size=n)  # 20 devices
    exam_types = rng.choice(
        ["abdominal", "cardiac", "obstetric", "vascular"],
        size=n
    )
    timestamps = pd.date_range("2025-01-01", periods=n, freq="min")

    chosen_patients = rng.choice(patient_ids, size=n)
    quality_scores = rng.normal(loc=0.8, scale=0.1, size=n).clip(0, 1)

    outcomes = []
    for pid, q in zip(chosen_patients, quality_scores):
        risk = risk_by_patient[pid]
        base_prob = 0.1
        if risk == "medium":
            base_prob = 0.2
        elif risk == "high":
            base_prob = 0.35
        if q < 0.6:
            base_prob += 0.15

        outcome = "follow_up_required" if rng.random() < base_prob else "normal"
        outcomes.append(outcome)

    df = pd.DataFrame({
        "exam_id": exam_ids,
        "patient_id": chosen_patients,
        "device_id": device_ids,
        "exam_type": exam_types,
        "exam_timestamp": timestamps,
        "image_quality_score": quality_scores,
        "outcome_label": outcomes,
    })
    return df

if __name__ == "__main__":
    df = generate_exams(n=30000)
    df.to_csv("data/raw/ultrasound_exams.csv", index=False)