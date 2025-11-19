from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .config import AGMConfigEntity
from .entities import SmoothingDecisionEntity


@dataclass(frozen=True)
class MultiScaleEma:
    short: float = 0.5
    medium: float = 0.8
    long: float = 0.95
    short_val: float = 0.0
    medium_val: float = 0.0
    long_val: float = 0.0

    def update(self, value: float) -> "MultiScaleEma":
        assert isinstance(value, (int, float)), "value must be numeric"
        # Create a new instance (immutability)
        return MultiScaleEma(
            short=self.short,
            medium=self.medium,
            long=self.long,
            short_val=(self.short * self.short_val + (1.0 - self.short) * float(value)),
            medium_val=(self.medium * self.medium_val + (1.0 - self.medium) * float(value)),
            long_val=(self.long * self.long_val + (1.0 - self.long) * float(value)),
        )

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_s": self.short_val,
            f"{prefix}_m": self.medium_val,
            f"{prefix}_l": self.long_val,
        }


class AdaptiveSmoothingController:
    """Adapt kernel smoothing parameters online based on stability signals.

    Policy (config-driven, no magic constants):
    - If td variance and disagreement are above configured minima, increase smoothing strength:
      alpha_step ↑ (clamped), k ↑ (clamped), tau ↓ (clamped).
    - If signals are well below minima, reduce smoothing strength to avoid bias.
    - Uses a small fractional change per update defined by `adapt_rate`.
    """

    def __init__(self, config: AGMConfigEntity):
        assert config is not None, "config required"
        self.config = config

    def adapt(
        self,
        *,
        cfg_run: AGMConfigEntity,
        step: int,
        ema_td_var: float,
        avg_disagreement: float,
        fep_value: Optional[float] = None,
    ) -> SmoothingDecisionEntity:
        ks = cfg_run.kernel_smoothing
        assert ks is not None, "kernel_smoothing required"

        # Exit if not enabled
        if not getattr(ks, "adapt_enabled", False):
            return SmoothingDecisionEntity(
                applied=False,
                alpha_step_used=ks.alpha_step_init,
                tau_clip=ks.trust_region_tau,
                gated_by_uncertainty=False,
                fep_value=fep_value,
                sigma_stats={},
            )

        # Decide direction based on thresholds
        td_high = float(ema_td_var) >= float(ks.td_var_min)
        dis_high = float(avg_disagreement) >= float(ks.disagreement_min)

        rate = float(getattr(ks, "adapt_rate", 0.05))

        new_alpha = float(ks.alpha_step_init)
        new_tau = float(ks.trust_region_tau)
        new_k = int(ks.k)

        if td_high and dis_high:
            # Increase smoothing
            new_alpha = min(float(getattr(ks, "alpha_step_max", ks.alpha_step_init)), new_alpha * (1.0 + rate))
            new_tau = max(float(getattr(ks, "tau_min", 0.0)), new_tau * (1.0 - rate))
            new_k = min(int(getattr(ks, "k_max", ks.k)), new_k + max(1, int(new_k * rate)))
        elif (float(ema_td_var) < (0.5 * float(ks.td_var_min))) and (float(avg_disagreement) < (0.5 * float(ks.disagreement_min) if ks.disagreement_min > 0 else 0.0)):
            # Reduce smoothing
            new_alpha = max(float(getattr(ks, "alpha_step_min", 0.0)), new_alpha * (1.0 - rate))
            new_tau = min(float(getattr(ks, "tau_max", new_tau)), new_tau * (1.0 + rate))
            new_k = max(int(getattr(ks, "k_min", 1)), new_k - max(1, int(new_k * rate)))

        # Ensure k respects batch constraint; caller must ensure batch >= k+1
        if new_k < 1:
            new_k = 1

        # Apply changes immutably via object.__setattr__
        if new_alpha != ks.alpha_step_init:
            object.__setattr__(cfg_run.kernel_smoothing, "alpha_step_init", float(new_alpha))
        if new_tau != ks.trust_region_tau:
            object.__setattr__(cfg_run.kernel_smoothing, "trust_region_tau", float(new_tau))
        if new_k != ks.k:
            object.__setattr__(cfg_run.kernel_smoothing, "k", int(new_k))

        return SmoothingDecisionEntity(
            applied=True,
            alpha_step_used=float(new_alpha),
            tau_clip=float(new_tau),
            gated_by_uncertainty=True,
            fep_value=fep_value,
            sigma_stats={"k": float(new_k)},
        )


def adapt_learning_rate(*, cfg_run: AGMConfigEntity, stability: float) -> float:
    """Adapt learning rate based on stability proxy in [0, 1].

    Higher stability → slightly increase LR within [lr_min, lr_max].
    Lower stability → decrease LR.
    """
    tr = cfg_run.training
    lr = tr.learning_rate
    rate = tr.lr_adapt_rate
    if stability >= 0.5:
        lr = min(tr.lr_max, lr * (1.0 + rate * (stability - 0.5) * 2.0))
    else:
        lr = max(tr.lr_min, lr * (1.0 - rate * (0.5 - stability) * 2.0))
    object.__setattr__(cfg_run.training, "learning_rate", float(lr))
    return float(lr)


