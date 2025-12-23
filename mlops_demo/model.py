"""Model utilities for training, evaluating, and persisting the classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@dataclass
class TrainResult:
    model_path: Path
    metrics_path: Path
    metrics: Dict[str, float]


class ModelTrainer:
    """Encapsulates model training and evaluation logic."""

    def __init__(self, artifact_dir: Path = Path("artifacts")) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        model = LogisticRegression(max_iter=200, multi_class="auto")
        model.fit(x_train, y_train)
        return model

    def evaluate(
        self, model: LogisticRegression, x_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        preds = model.predict(x_test)
        f1 = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()
        return {
            "f1_macro": float(f1),
            "accuracy": float(report["accuracy"]),
            "confusion_matrix": cm,
        }

    def save_model(self, model: LogisticRegression) -> Path:
        model_path = self.artifact_dir / "model.joblib"
        joblib.dump(model, model_path)
        return model_path

    def save_metrics(self, metrics: Dict[str, float]) -> Path:
        metrics_path = self.artifact_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return metrics_path

    def load_model(self) -> LogisticRegression:
        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Model artifact not found. Run the training pipeline first.")
        return joblib.load(model_path)

    def run(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> TrainResult:
        model = self.train(x_train, y_train)
        metrics = self.evaluate(model, x_test, y_test)
        model_path = self.save_model(model)
        metrics_path = self.save_metrics(metrics)
        return TrainResult(model_path=model_path, metrics_path=metrics_path, metrics=metrics)
