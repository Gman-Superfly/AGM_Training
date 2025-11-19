from agmlib.agm.wiring import ControllerSuite


def test_controller_suite_outputs_keys():
    suite = ControllerSuite()
    tel = {
        "step": 1,
        "agm": {
            "arithmetic_history": [1.0, 0.9, 0.8],
            "harmonic_history": [0.5, 0.55, 0.6],
        },
        "rl": {
            "td": {"mean": 0.0, "var": 0.01},
            "q": {"rel_change": 0.0},
            "reward": {"train_mean": 0.0, "eval_mean": 0.0},
        },
        "muon_clip": {"active": False, "clip_rate": 0.0},
    }
    out = suite.update(tel)
    assert set(["step", "convergence_rate", "phase", "phase_config", "hyperparams", "hierarchical", "early_stop", "uncertainty"]).issubset(out.keys())


