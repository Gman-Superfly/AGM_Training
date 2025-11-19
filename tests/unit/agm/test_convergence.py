from agmlib.agm.convergence import AdaptiveAGMController, ConvergenceConfig


def test_convergence_rate_and_regime():
    ctrl = AdaptiveAGMController(ConvergenceConfig(regime_window=5))
    # Create improving series where A and H approach each other
    arithmetic = [1.0, 0.9, 0.8, 0.7, 0.6]
    harmonic = [0.0, 0.1, 0.2, 0.3, 0.4]
    rate = ctrl.update(arithmetic, harmonic)
    assert 0.0 <= rate <= 1.0
    # Fill enough steps to trigger regime detection
    for _ in range(4):
        ctrl.update(arithmetic, harmonic)
    regime = ctrl.detect_training_regime()
    assert regime in {"stable_learning", "normal_learning", "exploration"}


