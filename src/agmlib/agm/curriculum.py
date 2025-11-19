from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping

from pydantic import BaseModel, Field, ConfigDict


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    difficulty_levels: List[str] = Field(default_factory=lambda: ["easy", "medium", "hard", "expert"])
    stability_requirement: float = Field(ge=0.0, le=1.0, default=0.7)
    mastery_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    drop_threshold: float = Field(ge=0.0, le=1.0, default=0.3)
    intermediate_band_low: float = Field(ge=0.0, le=1.0, default=0.3)
    intermediate_band_high: float = Field(ge=0.0, le=1.0, default=0.6)


class AGMCurriculumController:
    """Control curriculum progression using AGM convergence signals and performance metrics."""

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        self.cfg = config or CurriculumConfig()
        self.current_level = 0
        self.progression_history: List[Dict[str, Any]] = []

    def update_curriculum(
        self,
        *,
        agm_convergence_rate: float,
        success_rate: float,
        performance_metrics: Mapping[str, float],
    ) -> str:
        assert 0.0 <= agm_convergence_rate <= 1.0, "convergence rate must be in [0,1]"
        assert 0.0 <= success_rate <= 1.0, "success rate must be in [0,1]"

        action = "maintain_difficulty"
        cfg = self.cfg

        # Promote: fast/stable convergence + mastery
        if agm_convergence_rate > cfg.stability_requirement and success_rate > cfg.mastery_threshold:
            if self.current_level < len(cfg.difficulty_levels) - 1:
                self.current_level += 1
                action = "increase_difficulty"
        # Demote: slow convergence + low success
        elif agm_convergence_rate < cfg.drop_threshold and success_rate < 0.4:
            if self.current_level > 0:
                self.current_level -= 1
                action = "decrease_difficulty"
        # Maintain within intermediate band
        elif cfg.intermediate_band_low <= agm_convergence_rate <= cfg.intermediate_band_high and success_rate > 0.5:
            action = "maintain_difficulty"
        # Consolidate if very high performance but unstable convergence
        elif success_rate > 0.9 and agm_convergence_rate < 0.5:
            action = "consolidate_current_level"

        # Record decision
        self.progression_history.append(
            {
                "timestamp": len(self.progression_history),
                "agm_rate": agm_convergence_rate,
                "success_rate": success_rate,
                "action": action,
                "level": self.current_level,
            }
        )
        return action

    def get_curriculum_state(self) -> Dict[str, Any]:
        level = self.current_level
        levels = self.cfg.difficulty_levels
        return {
            "current_level": level,
            "current_difficulty": levels[level],
            "total_levels": len(levels),
            "progression_rate": self._compute_progression_rate(),
            "stability_trend": self._compute_stability_trend(),
        }

    def _compute_progression_rate(self) -> float:
        if len(self.progression_history) < 10:
            return 0.0
        last = self.progression_history[-10:]
        inc = sum(1 for e in last if e["action"] == "increase_difficulty")
        dec = sum(1 for e in last if e["action"] == "decrease_difficulty")
        return float((inc - dec) / 10.0)

    def _compute_stability_trend(self) -> str:
        if len(self.progression_history) < 5:
            return "insufficient_data"
        rates = [e["agm_rate"] for e in self.progression_history[-5:]]
        n = len(rates)
        sx = (n - 1) * n / 2.0
        sxx = (n - 1) * n * (2 * n - 1) / 6.0
        sy = sum(rates)
        sxy = sum(i * v for i, v in enumerate(rates))
        denom = n * sxx - sx * sx
        slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
        if slope > 0.05:
            return "improving_stability"
        if slope < -0.05:
            return "decreasing_stability"
        return "stable"


