from agmlib.agm.uncertainty import UncertaintyGatingConfig, apply_uncertainty_gating
from agmlib.agm.hparam import BaseHyperparams


def test_uncertainty_gating_reverts_to_base():
    base = BaseHyperparams(learning_rate=1e-3, epsilon=0.1, batch_size=256, agm_iterations=5)
    cfg = UncertaintyGatingConfig(enabled=True, mode="total", threshold=0.1, revert_to_base=True)
    decisions = {
        "step": 1,
        "hyperparams": {"learning_rate": 0.002, "epsilon": 0.2, "batch_size": 512, "agm_iterations": 3},
    }
    out = apply_uncertainty_gating(decisions=decisions, epistemic=0.2, aleatoric=0.2, base=base, config=cfg)
    assert out["uncertainty_gated"] is True
    hp = out["hyperparams"]
    assert hp["learning_rate"] == base.learning_rate
    assert hp["epsilon"] == base.epsilon
    assert hp["batch_size"] == base.batch_size


