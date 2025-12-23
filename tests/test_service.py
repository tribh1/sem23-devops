from fastapi.testclient import TestClient

from mlops_demo.service import app, load_or_train_model


def test_predict_endpoint_returns_class():
    load_or_train_model()
    client = TestClient(app)

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert 0 <= body["confidence"] <= 1


def test_metrics_endpoint_works_after_prediction():
    client = TestClient(app)
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    client.post("/predict", json=payload)

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "total_predictions" in metrics.json()
