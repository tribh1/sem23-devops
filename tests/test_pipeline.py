import json

from mlops_demo.pipeline import TrainingPipeline


def test_training_pipeline_creates_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    pipeline = TrainingPipeline(artifact_dir=artifact_dir)
    result = pipeline.run()

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert (artifact_dir / "run_config.json").exists()


def test_training_pipeline_metrics_structure(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    pipeline = TrainingPipeline(artifact_dir=artifact_dir)
    result = pipeline.run()

    with result.metrics_path.open() as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1
