from __future__ import annotations

from typing import Any, Dict

from agmlib.adaptation import MultiScaleEma


class MultiScaleAGMFramework:
    """Maintain multi-scale trackers for AGM metrics and compute hierarchical suggestions."""

    def __init__(self) -> None:
        # Independent trackers (weights chosen to roughly reflect short/med/long)
        self.conv = MultiScaleEma(short=0.6, medium=0.85, long=0.97)
        self.spread = MultiScaleEma(short=0.6, medium=0.85, long=0.97)

    def update_all_trackers(self, agm_metrics: Dict[str, Any]) -> None:
        """Update EMA trackers from a metrics dict with 'convergence_rate' and 'mean_spread'."""
        assert isinstance(agm_metrics, dict), "agm_metrics must be a dict"
        conv = float(agm_metrics.get("convergence_rate", 0.5))
        spr = float(agm_metrics.get("mean_spread", 0.0))
        self.conv = self.conv.update(conv)
        self.spread = self.spread.update(spr)

    def compute_hierarchical_adaptation(self) -> Dict[str, str]:
        """Return immediate/tactical/strategic suggestions based on multi-scale signals."""
        out: Dict[str, str] = {}
        # Immediate (short-term instability or excellent convergence)
        if self.spread.short_val > 0.4:
            out["immediate"] = "emergency_conservative_mode"
        elif self.conv.short_val > 0.9:
            out["immediate"] = "accelerate_slightly"
        else:
            out["immediate"] = "continue_current"

        # Tactical (trend via short vs medium)
        if self.conv.short_val - self.conv.medium_val > 0.02:
            out["tactical"] = "increase_exploration"
        elif self.conv.medium_val - self.conv.short_val > 0.02:
            out["tactical"] = "increase_stability"
        else:
            out["tactical"] = "maintain_course"

        # Strategic (regime via long-term)
        if self.conv.long_val > 0.8:
            out["strategic"] = "maintain_strategy"
        elif self.conv.long_val < 0.4:
            out["strategic"] = "reset_to_checkpoint"
        else:
            out["strategic"] = "standard_operation"

        return out


