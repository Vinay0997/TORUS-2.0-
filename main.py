from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import os
import glob
import pandas as pd

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MODEL_DIR = "models"

def find_model_path() -> str:
    """
    Automatically locate the saved model file in the models/ directory.
    Looks for any file matching followup_prediction_*.pkl
    """
    candidates = glob.glob(os.path.join(MODEL_DIR, "followup_prediction_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in '{MODEL_DIR}'. "
            "Run 05_train_followup_prediction.ipynb first to generate one."
        )
    return candidates[0]

MODEL_PATH = find_model_path()
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------

app = FastAPI(
    title="TORUS-2.0 Follow-Up Prediction API",
    description="Predicts whether an ultrasound exam requires clinical follow-up.",
    version="1.0.0",
)

# ---------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------

class ExamInput(BaseModel):
    age: int = Field(..., ge=0, le=120, example=45)
    sex: Literal["M", "F"] = Field(..., example="F")
    risk_category: Literal["low", "medium", "high"] = Field(..., example="medium")
    device_id: str = Field(..., example="DEV-002")
    exam_type: Literal["abdominal", "cardiac", "obstetric", "vascular"] = Field(
        ..., example="cardiac"
    )
    image_quality_score: float = Field(..., ge=0.0, le=1.0, example=0.82)


class PredictionOutput(BaseModel):
    follow_up_required: bool
    follow_up_probability: float
    model_used: str


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "TORUS-2.0 Follow-Up Prediction API is running.",
        "model_file": os.path.basename(MODEL_PATH),
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "model_file": os.path.basename(MODEL_PATH)}


@app.post("/predict", response_model=PredictionOutput)
def predict(exam: ExamInput):
    try:
        input_df = pd.DataFrame([{
            "age": exam.age,
            "sex": exam.sex,
            "risk_category": exam.risk_category,
            "device_id": exam.device_id,
            "exam_type": exam.exam_type,
            "image_quality_score": exam.image_quality_score,
        }])

        proba = model.predict_proba(input_df)[:, 1][0]
        prediction = bool(model.predict(input_df)[0])

        return PredictionOutput(
            follow_up_required=prediction,
            follow_up_probability=round(float(proba), 4),
            model_used=os.path.basename(MODEL_PATH),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------
# Run locally with: uvicorn main:app --reload
# ---------------------------------------------------------
