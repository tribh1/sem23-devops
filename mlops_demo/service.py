"""FastAPI service for inference, monitoring, and health checks."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from mlops_demo.data import IrisDataLoader
from mlops_demo.model import ModelTrainer
from mlops_demo.monitoring import monitoring_state

app = FastAPI(title="Iris MLOps Demo")


class InferencePayload(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ])


class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float


ARTIFACT_DIR = Path("artifacts")
trainer = ModelTrainer(artifact_dir=ARTIFACT_DIR)


def load_or_train_model() -> LogisticRegression:
    try:
        return trainer.load_model()
    except FileNotFoundError:
        dataset = IrisDataLoader().load()
        result = trainer.run(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test)
        return trainer.load_model()


@app.on_event("startup")
def _load_model() -> None:
    global model
    model = load_or_train_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: InferencePayload) -> PredictionResponse:
    if "model" not in globals():
        raise HTTPException(status_code=500, detail="Model not loaded")

    arr = payload.to_array().reshape(1, -1)
    probs = model.predict_proba(arr)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    monitoring_state.record(predicted_class, confidence)
    return PredictionResponse(predicted_class=predicted_class, confidence=confidence)


@app.get("/metrics")
def metrics() -> dict:
    if "model" not in globals():
        raise HTTPException(status_code=500, detail="Model not loaded")
    return monitoring_state.summary()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
