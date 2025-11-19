from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict

from .hparam import BaseHyperparams


class UncertaintyGatingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False)
    # Mode: total = 0.4*spread + 0.4*(1-conv) + 0.2*osc, or specific component
    mode: str = Field(default="total")  # "total" | "epistemic" | "aleatoric"
    threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    # Policy when gated: revert to base hyperparams (conservative)
    revert_to_base: bool = Field(default=True)


def apply_uncertainty_gating(
    *,
    decisions: Dict[str, Any],
    epistemic: float,
    aleatoric: float,
    base: BaseHyperparams,
    config: Optional[UncertaintyGatingConfig] = None,
) -> Dict[str, Any]:
    """Apply uncertainty gating to controller decisions.

    - If enabled and uncertainty exceeds threshold, return conservative hyperparams.
    - Adds 'uncertainty_gated' flag to decisions.
    """
    cfg = config or UncertaintyGatingConfig()
    out = dict(decisions)
    if not cfg.enabled:
        out["uncertainty_gated"] = False
        return out

    total = 0.4 * float(epistemic) + 0.6 * float(aleatoric)  # slightly tilt towards data noise
    if cfg.mode == "epistemic":
        score = float(epistemic)
    elif cfg.mode == "aleatoric":
        score = float(aleatoric)
    else:
        score = total

    if score > cfg.threshold:
        out["uncertainty_gated"] = True
        if cfg.revert_to_base:
            out["hyperparams"] = {
                "learning_rate": float(base.learning_rate),
                "epsilon": float(base.epsilon),
                "batch_size": float(base.batch_size),
                "agm_iterations": float(base.agm_iterations),
            }
        return out
    out["uncertainty_gated"] = False
    return out


