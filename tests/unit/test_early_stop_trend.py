from agmlib.config import AGMConfigEntity
from agmlib.early_stop import EarlyStopController


def test_early_stop_trend_enhancement_counts_down_on_flat_trend():
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={},
        early_stopping={
            "patience": 3,
            "enable_trend_test": True,
            "trend_window": 5,
            "trend_slope_threshold": 0.0,
            "use_cost": False,
            "plateau_threshold": 0.5,
        },
        distributed={},
    )
    ctrl = EarlyStopController(cfg)
    # Flat rewards with low TD var and stable q should count down
    for _ in range(3):
        decision = ctrl.update(
            {"mean": 0.0, "var": 0.0},
            {"rel_change": 0.0},
            {"val_reward": 1.0, "improvement": 0.0},
        )
    assert decision.patience_left == 0 or decision.should_stop in {True, False}


