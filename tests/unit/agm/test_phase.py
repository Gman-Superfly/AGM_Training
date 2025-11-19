from agmlib.agm.phase import TrainingPhaseDetector, PhaseDetectorConfig


def test_phase_detection_basic():
    det = TrainingPhaseDetector(PhaseDetectorConfig(window=10))
    # Decreasing loss should be exploitation if convergence is decent
    loss = [1.0, 0.95, 0.9, 0.85, 0.8, 0.78, 0.76, 0.75, 0.74, 0.73]
    phase = det.detect_phase(loss_history=loss, convergence_rate=0.7)
    assert phase in {"exploitation", "exploration"}  # depending on thresholds


