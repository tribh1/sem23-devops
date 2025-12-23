"""End-to-end MLOps training pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from mlops_demo.data import IrisDataLoader
from mlops_demo.model import ModelTrainer, TrainResult


class TrainingPipeline:
    def __init__(self, artifact_dir: Path = Path("artifacts")) -> None:
        self.artifact_dir = artifact_dir
        self.data_loader = IrisDataLoader()
        self.trainer = ModelTrainer(artifact_dir=artifact_dir)

    def run(self) -> TrainResult:
        dataset = self.data_loader.load()
        result = self.trainer.run(
            dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test
        )
        self._save_config()
        return result

    def _save_config(self) -> None:
        config_path = self.artifact_dir / "run_config.json"
        config = {
            "data": {
                "test_size": self.data_loader.test_size,
                "random_state": self.data_loader.random_state,
            },
            "model": {"type": "LogisticRegression", "max_iter": 200},
        }
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
