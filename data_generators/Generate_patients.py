"""
generate_patients.py
Generates synthetic patient records for the TORUS-2.0 project.
Saves output to data/raw/patients.csv
"""

import pandas as pd
import numpy as np

def generate_patients(n_patients: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    patients_df = pd.DataFrame({
        "patient_id": range(1, n_patients + 1),
        "age": np.random.randint(18, 85, n_patients),
        "sex": np.random.choice(["M", "F"], n_patients),
        "risk_category": np.random.choice(
            ["low", "medium", "high"], n_patients, p=[0.5, 0.35, 0.15]
        ),
    })
    return patients_df


if __name__ == "__main__":
    import os

    df = generate_patients(n_patients=300)

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/patients.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} patient records -> {output_path}")
