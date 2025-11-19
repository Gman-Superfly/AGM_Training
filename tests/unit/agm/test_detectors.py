from agmlib.agm.detectors import AGMConvergenceDetector, ConvergenceStopConfig, AGMUncertaintyEstimator


def test_convergence_detector_does_not_crash_and_returns_tuple():
    det = AGMConvergenceDetector(ConvergenceStopConfig(patience=5))
    for i in range(25):
        should, reason = det.should_stop_training(
            agm_metrics={"convergence_rate": 0.7},
            performance_metrics={"primary_metric": float(i)},
        )
        assert isinstance(should, bool) and isinstance(reason, str)


def test_uncertainty_estimator_bounds():
    ue = AGMUncertaintyEstimator()
    agm_state = {"arithmetic_history": [1.0, 0.9, 0.8, 0.7], "harmonic_history": [0.5, 0.55, 0.6, 0.65]}
    epi = ue.estimate_epistemic_uncertainty(agm_state)
    alea = ue.estimate_aleatoric_uncertainty(agm_state, performance_variance=0.01)
    assert 0.0 <= epi <= 1.0
    assert 0.0 <= alea <= 1.0


