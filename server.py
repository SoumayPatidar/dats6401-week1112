# server.py : FastAPI endpoint that predicts F1 podium finishes
# run with: uvicorn server:app --reload
# then open http://127.0.0.1:8000/docs
# Soumay Patidar | DATS 6401

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="F1 Podium Prediction API",
    description="Predicts whether a driver will finish on the podium based on "
                "grid position, constructor, circuit, and race info.",
)

#  load the model once at startup
model_path = Path(__file__).resolve().parent / "model.joblib"
bundle = joblib.load(model_path)
model = bundle["model"]
FEATURES = bundle["feature_names"]
le_constructor = bundle["le_constructor"]
le_circuit = bundle["le_circuit"]
constructor_stats = bundle["constructor_stats"]
KNOWN_CONSTRUCTORS = bundle["known_constructors"]
KNOWN_CIRCUITS = bundle["known_circuits"]


# ── request / response schemas ──
class RaceEntry(BaseModel):
    """info about a driver's race entry"""
    grid: int = Field(..., ge=1, le=30, description="Starting grid position (1-30)")
    constructor: str = Field(..., description="Constructor name, e.g. 'red_bull', 'mercedes', 'ferrari'")
    circuit_id: int = Field(..., ge=1, description="Circuit ID from the Ergast database")
    race_round: int = Field(..., ge=1, le=25, description="Round number in the season (1-25)")
    laps: int = Field(57, ge=1, le=100, description="Number of laps in the race")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "grid": 1,
                    "constructor": "red_bull",
                    "circuit_id": 9,
                    "race_round": 1,
                    "laps": 57,
                }
            ]
        }
    }


class PodiumPrediction(BaseModel):
    """what the server sends back"""
    prediction: str
    podium_probability: float
    no_podium_probability: float
    grid: int
    constructor: str


# ── endpoints ──
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "features": FEATURES,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "known_constructors": KNOWN_CONSTRUCTORS,
        "known_circuits": [str(c) for c in KNOWN_CIRCUITS],
        "training_era": "2000-2024",
    }


@app.get("/constructors")
def list_constructors():
    """list all constructors the model knows about"""
    return {"constructors": KNOWN_CONSTRUCTORS}


@app.post("/predict", response_model=PodiumPrediction)
def predict(entry: RaceEntry):
    """predict whether this entry will finish on the podium"""

    # check if we know this constructor
    if entry.constructor not in KNOWN_CONSTRUCTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown constructor '{entry.constructor}'. "
                   f"Use /constructors to see valid options. "
                   f"Some common ones: red_bull, mercedes, ferrari, mclaren, williams",
        )

    # check if we know this circuit
    circuit_str = str(entry.circuit_id)
    if circuit_str not in KNOWN_CIRCUITS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown circuit_id {entry.circuit_id}. "
                   f"Use /model-info to see known circuits.",
        )

    # build the feature row
    constructor_rate = constructor_stats.get(entry.constructor, 0.0)
    constructor_enc = int(le_constructor.transform([entry.constructor])[0])
    circuit_enc = int(le_circuit.transform([circuit_str])[0])

    row = pd.DataFrame([{
        "grid": entry.grid,
        "constructor_podium_rate": constructor_rate,
        "constructor_encoded": constructor_enc,
        "laps": entry.laps,
        "circuit_encoded": circuit_enc,
        "round": entry.race_round,
    }], columns=FEATURES)

    proba = model.predict_proba(row)[0]
    podium_prob = float(proba[1])
    no_podium_prob = float(proba[0])

    prediction = "Podium" if podium_prob > 0.5 else "No podium"

    return {
        "prediction": prediction,
        "podium_probability": round(podium_prob, 4),
        "no_podium_probability": round(no_podium_prob, 4),
        "grid": entry.grid,
        "constructor": entry.constructor,
    }
