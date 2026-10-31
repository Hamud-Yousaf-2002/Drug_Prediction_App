from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "drug_prediction_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Drug Prediction API")


class Patient(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Sex: Literal["M", "F"]
    BP: Literal["LOW", "NORMAL", "HIGH"]
    Cholesterol: Literal["NORMAL", "HIGH"]
    Na_to_K: float = Field(..., gt=0)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(df)[0].item()
    probs = model.predict_proba(df)[0]
    return {
        "prediction": prediction,
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)},
    }
