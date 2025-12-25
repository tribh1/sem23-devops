import json
import logging
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

MODEL_NAME = os.getenv("MODEL_NAME", "sentiment140-logreg")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # Test/Staging/Production

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

app = FastAPI(title="Sentiment140 Serving")
logger = logging.getLogger("serving")

class Req(BaseModel):
    text: str

def load_model():
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(uri)

model = None

@app.on_event("startup")
def startup():
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    model = load_model()
    logger.info(json.dumps({
        "service": "serving",
        "event": "startup",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE
    }))

@app.get("/health")
def health():
    logger.info(json.dumps({
        "service": "serving",
        "event": "health_check",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE
    }))
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(req: Req):
    global model
    t0 = time.time()
    pred = int(model.predict([req.text])[0])
    latency = int((time.time() - t0) * 1000)
    logger.info(json.dumps({
        "service":"serving",
        "event":"prediction",
        "model":MODEL_NAME,
        "stage":MODEL_STAGE,
        "latency_ms":latency,
        "prediction":pred
    }))
    return {"prediction": pred, "latency_ms": latency}
