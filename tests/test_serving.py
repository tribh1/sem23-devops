from fastapi.testclient import TestClient

import src.serving.app as serving_app


class DummyModel:
    def predict(self, items):
        return [1 for _ in items]


def test_health_endpoint_reports_model_and_stage():
    serving_app.app.router.on_startup.clear()
    serving_app.model = DummyModel()

    client = TestClient(serving_app.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == serving_app.MODEL_NAME
    assert payload["stage"] == serving_app.MODEL_STAGE


def test_predict_endpoint_returns_prediction_and_latency():
    serving_app.app.router.on_startup.clear()
    serving_app.model = DummyModel()

    client = TestClient(serving_app.app)
    response = client.post("/predict", json={"text": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1
    assert isinstance(payload["latency_ms"], int)
