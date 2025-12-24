import os, time, json
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

MODEL_NAME = os.getenv("MODEL_NAME", "sentiment140-logreg")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # Test/Staging/Production

app = FastAPI(title="Sentiment140 Serving")

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

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(req: Req):
    global model
    t0 = time.time()
    pred = int(model.predict([req.text])[0])
    latency = int((time.time() - t0) * 1000)
    print(json.dumps({
        "service":"serving",
        "event":"prediction",
        "model":MODEL_NAME,
        "stage":MODEL_STAGE,
        "latency_ms":latency,
        "prediction":pred
    }))
    return {"prediction": pred, "latency_ms": latency}
