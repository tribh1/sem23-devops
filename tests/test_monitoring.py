from mlops_demo.monitoring import MonitoringState


def test_summary_empty_state_returns_defaults():
    state = MonitoringState()
    summary = state.summary()

    assert summary["total_predictions"] == 0
    assert summary["avg_confidence"] is None
    assert summary["class_distribution"] == {}
    assert summary["drift_score"] is None


def test_summary_after_records_updates_metrics():
    state = MonitoringState()
    state.record(predicted_class=0, confidence=0.9)
    state.record(predicted_class=1, confidence=0.8)
    state.record(predicted_class=1, confidence=0.6)

    summary = state.summary()

    assert summary["total_predictions"] == 3
    assert 0.7 <= summary["avg_confidence"] <= 0.9
    # class 1 should appear twice, class 0 once
    assert summary["class_distribution"] == {0: 1, 1: 2}
    # drift score should be a non-negative float
    assert isinstance(summary["drift_score"], float)
    assert summary["drift_score"] >= 0
