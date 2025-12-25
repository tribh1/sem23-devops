import numpy as np
from src.train.train_register_mlflow import metrics


def test_metrics_binary_classification():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    result = metrics(y_true, y_pred)

    assert result["accuracy"] == 0.75
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == 2 * (1.0 * 0.5) / (1.0 + 0.5)
