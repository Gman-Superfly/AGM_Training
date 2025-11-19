from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from pydantic import BaseModel, Field, ConfigDict


class PhaseDetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = Field(ge=5, default=20)
    improving_slope: float = Field(default=-0.01, description="Slope < this indicates improving loss")
    plateau_slope_abs: float = Field(ge=0.0, default=0.001, description="|slope| <= this -> plateau-ish")
    low_variance: float = Field(ge=0.0, default=0.01)
    unstable_convergence_threshold: float = Field(ge=0.0, le=1.0, default=0.3)
    stable_convergence_threshold: float = Field(ge=0.0, le=1.0, default=0.6)


def _linear_slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    # Simple least squares slope with x=0..n-1
    sx = (n - 1) * n / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = sum(values)
    sxy = sum(i * v for i, v in enumerate(values))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    return float(slope)


def _variance(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / float(n)
    return float(sum((v - m) * (v - m) for v in values) / float(n))


class TrainingPhaseDetector:
    """Detect training phase from recent loss trend/variance and convergence rate."""

    def __init__(self, config: PhaseDetectorConfig | None = None) -> None:
        self.config = config or PhaseDetectorConfig()

    def detect_phase(self, *, loss_history: Sequence[float], convergence_rate: float) -> str:
        c = self.config
        if len(loss_history) < c.window:
            return "exploration"
        recent = list(loss_history[-c.window :])
        slope = _linear_slope(recent)
        var = _variance(recent)
        if slope < c.improving_slope and convergence_rate > c.stable_convergence_threshold:
            return "exploitation"
        if var > c.low_variance or convergence_rate < c.unstable_convergence_threshold:
            return "instability"
        if abs(slope) <= c.plateau_slope_abs and var <= c.low_variance:
            return "plateau"
        if slope < 0.0:
            return "exploitation"
        return "exploration"


class TrainingPhaseAdaptiveAGM:
    """Select AGM strategy based on the detected phase."""

    def __init__(self, detector: TrainingPhaseDetector | None = None) -> None:
        self.detector = detector or TrainingPhaseDetector()
        self.strategy_history: list[tuple[str, str]] = []

    def select_adaptive_strategy(
        self,
        *,
        loss_history: Sequence[float],
        performance_metrics: Dict[str, Any],
        agm_convergence_rate: float,
    ) -> Tuple[str, Dict[str, Any]]:
        phase = self.detector.detect_phase(loss_history=loss_history, convergence_rate=agm_convergence_rate)
        strategy: str
        cfg: Dict[str, Any] = {}
        if phase == "exploration":
            strategy = "arithmetic_weighted_agm"
            cfg = {"arithmetic_weight": 0.8, "agm_iterations": 3, "convergence_threshold": 1e-3}
        elif phase == "exploitation":
            strategy = "standard_agm"
            cfg = {"agm_iterations": 5, "convergence_threshold": 1e-6}
        elif phase == "instability":
            strategy = "harmonic_weighted_agm"
            cfg = {"harmonic_weight": 0.8, "agm_iterations": 8, "convergence_threshold": 1e-5}
        elif phase == "plateau":
            strategy = "oscillating_agm"
            cfg = {"oscillation_period": 10, "exploration_boost": 0.3}
        else:  # normal_learning or unknown
            strategy = "standard_agm"
            cfg = {"agm_iterations": 5}
        self.strategy_history.append((phase, strategy))
        return strategy, cfg


