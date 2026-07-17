# data_generators/generate_patients.py
import pandas as pd
import numpy as np

def generate_patients(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    patient_ids = np.arange(1, n+1)

    ages = rng.integers(18, 90, size=n)
    sexes = rng.choice(["M", "F"], size=n)
    risk_categories = rng.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1], size=n)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "sex": sexes,
        "risk_category": risk_categories,
    })
    return df

if __name__ == "__main__":
    df = generate_patients(n=2000)
    df.to_csv("data/raw/patients.csv", index=False)