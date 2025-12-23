"""Simple in-memory monitoring utilities for the FastAPI service."""
from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class PredictionLog:
    predicted_class: int
    confidence: float


@dataclass
class MonitoringState:
    window_size: int = 200
    logs: Deque[PredictionLog] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, predicted_class: int, confidence: float) -> None:
        self.logs.append(PredictionLog(predicted_class, confidence))

    def summary(self) -> Dict[str, object]:
        if not self.logs:
            return {
                "total_predictions": 0,
                "avg_confidence": None,
                "class_distribution": {},
                "drift_score": None,
            }

        confidences = [log.confidence for log in self.logs]
        classes = [log.predicted_class for log in self.logs]
        drift_score = self._simple_drift_score(classes)
        return {
            "total_predictions": len(self.logs),
            "avg_confidence": float(statistics.mean(confidences)),
            "class_distribution": dict(Counter(classes)),
            "drift_score": drift_score,
        }

    def _simple_drift_score(self, classes: Iterable[int]) -> float:
        counts = Counter(classes)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probabilities = np.array([counts[c] / total for c in sorted(counts)])
        expected = np.full_like(probabilities, 1 / len(probabilities), dtype=float)
        drift = float(np.linalg.norm(probabilities - expected))
        return drift


monitoring_state = MonitoringState()
