"""
FastAPI prediction endpoint for thyroid disease classification.

Usage: uvicorn api.predict:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Thyroid Disease Classifier API", version="1.0")

# Load model on startup
models_dir = "models"
meta_files = sorted([f for f in os.listdir(models_dir) if f.startswith("metadata_")], reverse=True)
timestamp = meta_files[0].replace("metadata_", "").replace(".json", "")
model = joblib.load(f"{models_dir}/best_model_{timestamp}.pkl")
scaler = joblib.load(f"{models_dir}/scaler_{timestamp}.pkl")
label_encoder = joblib.load(f"{models_dir}/label_encoder_{timestamp}.pkl")
with open(f"{models_dir}/metadata_{timestamp}.json") as f:
    metadata = json.load(f)
features = metadata["features_selected"]


class PatientInput(BaseModel):
    TSH: float = 2.5
    T3: float = 1.8
    T4: float = 105.0
    T4U: float = 1.0
    age: float = 45.0
    sex: int = 0
    on_thyroxine: int = 0
    on_antithyroid: int = 0
    sick: int = 0
    pregnant: int = 0
    thyroid_surgery: int = 0
    goitre: int = 0


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    features_used: list


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["best_model"]}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    data = patient.model_dump()
    data["FTI"] = data["T4"] / (data["T4U"] + 0.01)

    for f in features:
        if f not in data:
            data[f] = 0

    input_df = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    input_scaled = scaler.transform(input_df)

    prediction_idx = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_idx])[0]

    return PredictionOutput(
        prediction=prediction,
        confidence=float(probabilities[prediction_idx]),
        probabilities={cls: float(p) for cls, p in zip(label_encoder.classes_, probabilities)},
        features_used=features,
    )
