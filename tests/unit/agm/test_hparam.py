from agmlib.agm.hparam import AGMHyperparameterAdapter, BaseHyperparams, HParamAdapterConfig


def test_hparam_adapter_suggestions_within_bounds():
    base = BaseHyperparams(learning_rate=1e-3, epsilon=0.1, batch_size=256, agm_iterations=5)
    cfg = HParamAdapterConfig()
    adapter = AGMHyperparameterAdapter(base_hyperparams=base, config=cfg)
    agm_state = {
        "agm": {
            "arithmetic_history": [1.0, 0.9, 0.8, 0.7, 0.6],
            "harmonic_history": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    }
    sugg = adapter.adapt_hyperparameters(agm_state=agm_state, performance_trend=0.0)
    assert cfg.bounds.lr_min <= sugg["learning_rate"] <= cfg.bounds.lr_max
    assert cfg.bounds.eps_min <= sugg["epsilon"] <= cfg.bounds.eps_max
    assert cfg.bounds.batch_min <= sugg["batch_size"] <= cfg.bounds.batch_max
    assert cfg.bounds.agm_min <= sugg["agm_iterations"] <= cfg.bounds.agm_max


