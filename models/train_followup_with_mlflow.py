"""
train_followup_with_mlflow.py
TORUS-2.0: Step 3 (finish) - MLflow tracking for follow-up prediction.

Place this file at: models/train_followup_with_mlflow.py
Run it from repo root:  python models/train_followup_with_mlflow.py

Mirrors the exact pipeline structure from 05_train_followup_prediction.ipynb:
  - ColumnTransformer: OneHotEncoder on categorical_cols, passthrough numeric
  - log_reg_pipeline: LogisticRegression(max_iter=1000, class_weight="balanced")
  - rf_pipeline: RandomForestClassifier(n_estimators=200, max_depth=8,
                 random_state=42, class_weight="balanced")
  - Best model selected by F1 score (same logic as notebook)

Logs BOTH models as separate MLflow runs (not just the winner) so you get
a full side-by-side comparison in the MLflow UI, then saves the winning
model to models/followup_prediction_{best_model_name}.pkl exactly as the
notebook does.

NOTE: categorical_cols / numeric_cols are auto-detected below by dtype
(object -> categorical, numeric -> numeric), excluding the target and any
timestamp column. If your notebook defines these lists explicitly with a
different split, replace the CATEGORICAL_COLS / NUMERIC_COLS section below
with your exact lists.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DATA_PATH = "data/processed/modeling_dataset.csv"
TARGET_COL = "target"
DROP_COLS = ["outcome_label", "exam_timestamp", TARGET_COL]

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "torus-followup-prediction"


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    categorical_cols = [c for c in feature_cols if df[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, categorical_cols, numeric_cols


def build_pipelines(categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    log_reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        )),
    ])

    return log_reg_pipeline, rf_pipeline


def log_run(model_name, pipeline, params, X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        for k, v in params.items():
            mlflow.log_param(k, v)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"\n{model_name} Results")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")

        return f1, pipeline


def main():
    X, y, categorical_cols, numeric_cols = load_data()
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_reg_pipeline, rf_pipeline = build_pipelines(categorical_cols)

    f1_lr, log_reg_pipeline = log_run(
        "logistic_regression", log_reg_pipeline,
        {"model_type": "LogisticRegression", "max_iter": 1000, "class_weight": "balanced"},
        X_train, X_test, y_train, y_test,
    )

    f1_rf, rf_pipeline = log_run(
        "random_forest", rf_pipeline,
        {"model_type": "RandomForestClassifier", "n_estimators": 200,
         "max_depth": 8, "class_weight": "balanced"},
        X_train, X_test, y_train, y_test,
    )

    best_model = rf_pipeline if f1_rf >= f1_lr else log_reg_pipeline
    best_model_name = "random_forest" if best_model is rf_pipeline else "logistic_regression"
    print(f"\nBest model selected: {best_model_name}")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/followup_prediction_{best_model_name}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    print("\nDone. Launch the MLflow UI with:")
    print("  python -m mlflow ui --port 5001")
    print("Then open http://127.0.0.1:5001 to compare both runs.")


if __name__ == "__main__":
    main()
