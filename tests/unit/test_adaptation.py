from agmlib.config import AGMConfigEntity
from agmlib.adaptation import AdaptiveSmoothingController, adapt_learning_rate


def test_adaptive_smoothing_increase_and_clamps():
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={
            "enabled": True,
            "k": 16,
            "sigma_min": 1e-3,
            "sigma_max": 1.0,
            "alpha_step_init": 0.15,
            "alpha_step_schedule": "cosine",
            "alpha_step_decay_steps": 100,
            "dim_exponent": 0.5,
            "trust_region_tau": 0.1,
            "latent_dim": 8,
            "td_var_min": 0.01,
            "disagreement_min": 0.01,
            "adapt_enabled": True,
            "adapt_rate": 0.05,
            "alpha_step_min": 0.01,
            "alpha_step_max": 0.5,
            "tau_min": 0.01,
            "tau_max": 1.0,
            "k_min": 4,
            "k_max": 32,
        },
        early_stopping={},
        distributed={},
    )
    cfg_run = cfg.model_copy(deep=True)
    adaptor = AdaptiveSmoothingController(cfg_run)

    # High signals should increase smoothing (alpha/k up, tau down)
    decision = adaptor.adapt(
        cfg_run=cfg_run,
        step=0,
        ema_td_var=0.05,
        avg_disagreement=0.05,
        fep_value=None,
    )
    assert decision.applied is True
    # alpha increases by ~5%
    assert cfg_run.kernel_smoothing.alpha_step_init > 0.15
    # tau decreases by ~5%
    assert cfg_run.kernel_smoothing.trust_region_tau < 0.1
    # k increases at least by 1 but not over k_max
    assert cfg_run.kernel_smoothing.k >= 17
    assert cfg_run.kernel_smoothing.k <= cfg_run.kernel_smoothing.k_max


def test_adaptive_smoothing_disabled_returns_noop():
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={
            "enabled": True,
            "k": 8,
            "alpha_step_init": 0.2,
            "trust_region_tau": 0.1,
            "td_var_min": 0.0,
            "disagreement_min": 0.0,
            "adapt_enabled": False,
        },
        early_stopping={},
        distributed={},
    )
    cfg_run = cfg.model_copy(deep=True)
    adaptor = AdaptiveSmoothingController(cfg_run)
    decision = adaptor.adapt(
        cfg_run=cfg_run,
        step=10,
        ema_td_var=1.0,
        avg_disagreement=1.0,
        fep_value=None,
    )
    assert decision.applied is False
    # No mutation
    assert cfg_run.kernel_smoothing.alpha_step_init == cfg.kernel_smoothing.alpha_step_init
    assert cfg_run.kernel_smoothing.trust_region_tau == cfg.kernel_smoothing.trust_region_tau
    assert cfg_run.kernel_smoothing.k == cfg.kernel_smoothing.k


def test_adapt_learning_rate_bounds():
    cfg = AGMConfigEntity(
        replay={},
        kernel_smoothing={},
        early_stopping={},
        distributed={},
        training={
            "learning_rate": 1e-3,
            "lr_min": 1e-5,
            "lr_max": 5e-3,
            "lr_adapt_rate": 0.1,
        },
    )
    cfg_run = cfg.model_copy(deep=True)

    # High stability should increase LR but clamp at lr_max
    lr1 = adapt_learning_rate(cfg_run=cfg_run, stability=1.0)
    assert lr1 <= cfg_run.training.lr_max
    # Low stability should decrease LR but clamp at lr_min
    lr2 = adapt_learning_rate(cfg_run=cfg_run, stability=0.0)
    assert lr2 >= cfg_run.training.lr_min


