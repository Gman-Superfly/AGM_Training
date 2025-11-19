from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, List

from pydantic import BaseModel, Field, ConfigDict


def _to_float_list(x: Any) -> List[float]:
    try:
        import importlib

        torch = importlib.import_module("torch")  # type: ignore[assignment]
    except Exception:
        torch = None
    if torch is not None and hasattr(x, "detach"):
        try:
            return [float(v) for v in x.detach().cpu().numpy().ravel().tolist()]
        except Exception:
            pass
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):  # type: ignore[attr-defined]
            return [float(v) for v in x.ravel().tolist()]
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        return [float(x)]
    except Exception as e:
        raise AssertionError(f"cannot coerce to float list: {type(x)}") from e


class ConvergenceStopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    patience: int = Field(ge=1, default=50)
    min_improvement: float = Field(ge=0.0, default=1e-4)
    converge_mean_min: float = Field(ge=0.0, le=1.0, default=0.8)
    converge_var_max: float = Field(ge=0.0, default=0.01)


class AGMConvergenceDetector:
    """Detect training convergence/plateau/instability from AGM + performance patterns."""

    def __init__(self, config: ConvergenceStopConfig | None = None) -> None:
        self.cfg = config or ConvergenceStopConfig()
        self._agm_conv_hist: List[float] = []
        self._perf_hist: List[float] = []

    def _mean(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(sum(xs) / float(len(xs)))

    def _var(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        m = self._mean(xs)
        return float(sum((x - m) * (x - m) for x in xs) / float(len(xs)))

    def should_stop_training(self, *, agm_metrics: Mapping[str, Any], performance_metrics: Mapping[str, Any]) -> Tuple[bool, str]:
        # Update histories
        conv = float(agm_metrics.get("convergence_rate", 0.0))
        # Accept either RL reward or explicit primary metric
        perf = float(performance_metrics.get("primary_metric", performance_metrics.get("eval_reward", 0.0)))
        self._agm_conv_hist.append(conv)
        self._perf_hist.append(perf)
        # Evaluate rules
        if len(self._agm_conv_hist) >= 20:
            recent_conv = self._agm_conv_hist[-20:]
            if self._var(recent_conv) < 1e-6 and self._mean(recent_conv) > 0.99:
                return True, "agm_full_convergence"

        if len(self._agm_conv_hist) >= 20 and len(self._perf_hist) >= self.cfg.patience:
            recent_conv = self._agm_conv_hist[-20:]
            conv_converged = (self._mean(recent_conv) > self.cfg.converge_mean_min) and (self._var(recent_conv) < self.cfg.converge_var_max)
            if conv_converged:
                recent_perf = self._perf_hist[-self.cfg.patience :]
                perf_improvement = recent_perf[-1] - recent_perf[0]
                if abs(perf_improvement) < self.cfg.min_improvement:
                    return True, "agm_convergence_with_plateau"

        if len(self._agm_conv_hist) >= 30:
            recent = self._agm_conv_hist[-30:]
            if self._var(recent) > 0.1 or self._mean(recent) < 0.1:
                return True, "agm_detected_instability"

        if len(self._agm_conv_hist) >= 40 and len(self._perf_hist) >= 40:
            conv_recent = self._agm_conv_hist[-10:]
            perf_recent = self._perf_hist[-20:]
            mean_conv = self._mean(conv_recent)
            var_conv = self._var(conv_recent)
            # Simple slope estimation for performance
            n = len(perf_recent)
            sx = (n - 1) * n / 2.0
            sxx = (n - 1) * n * (2 * n - 1) / 6.0
            sy = sum(perf_recent)
            sxy = sum(i * v for i, v in enumerate(perf_recent))
            denom = n * sxx - sx * sx
            perf_slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
            if mean_conv > 0.85 and var_conv < 0.01 and self._mean(perf_recent[-10:]) > 0.8 and abs(perf_slope) < self.cfg.min_improvement / 10.0:
                return True, "agm_optimal_point_reached"

        return False, "continue_training"


class AGMUncertaintyEstimator:
    """Estimate epistemic and aleatoric uncertainty from AGM patterns."""

    def estimate_epistemic_uncertainty(self, agm_state: Mapping[str, Any]) -> float:
        a_hist = _to_float_list(agm_state.get("arithmetic_history", agm_state.get("agm", {}).get("arithmetic_history", [])))
        h_hist = _to_float_list(agm_state.get("harmonic_history", agm_state.get("agm", {}).get("harmonic_history", [])))
        # Spread
        spread = 0.0
        if a_hist and h_hist:
            a = a_hist[-1]
            h = h_hist[-1]
            spread = abs(a - h) / (abs(a) + abs(h) + 1e-8)
        # Convergence speed proxy
        if len(a_hist) >= 2 and len(h_hist) >= 2:
            gaps = [abs(a_hist[i] - h_hist[i]) for i in range(min(len(a_hist), len(h_hist)))]
            if gaps[0] > 1e-8:
                conv = max(0.0, (gaps[0] - gaps[-1]) / gaps[0])
            else:
                conv = 0.5
        else:
            conv = 0.5
        # Oscillation strength via variance of recent diffs
        recent_a = a_hist[-10:]
        recent_h = h_hist[-10:]
        diffa = [abs(recent_a[i] - recent_a[i - 1]) for i in range(1, len(recent_a))] if len(recent_a) >= 2 else []
        diffh = [abs(recent_h[i] - recent_h[i - 1]) for i in range(1, len(recent_h))] if len(recent_h) >= 2 else []
        def _var(xs: List[float]) -> float:
            if not xs:
                return 0.0
            m = sum(xs) / float(len(xs))
            return sum((x - m) * (x - m) for x in xs) / float(len(xs))
        osc = (_var(diffa) + _var(diffh)) / 2.0
        # Combine
        spread_u = min(1.0, spread * 10.0)
        conv_u = 1.0 - min(1.0, max(0.0, conv))
        osc_u = min(1.0, osc * 1000.0)
        total = 0.4 * spread_u + 0.4 * conv_u + 0.2 * osc_u
        return float(total)

    def estimate_aleatoric_uncertainty(self, agm_state: Mapping[str, Any], *, performance_variance: float) -> float:
        recent_a = _to_float_list(agm_state.get("arithmetic_history", agm_state.get("agm", {}).get("arithmetic_history", [])))[-10:]
        recent_h = _to_float_list(agm_state.get("harmonic_history", agm_state.get("agm", {}).get("harmonic_history", [])))[-10:]
        def _var(xs: List[float]) -> float:
            if not xs:
                return 0.0
            m = sum(xs) / float(len(xs))
            return sum((x - m) * (x - m) for x in xs) / float(len(xs))
        osc = 0.0
        if len(recent_a) >= 2 and len(recent_h) >= 2:
            diffa = [abs(recent_a[i] - recent_a[i - 1]) for i in range(1, len(recent_a))]
            diffh = [abs(recent_h[i] - recent_h[i - 1]) for i in range(1, len(recent_h))]
            osc = (_var(diffa) + _var(diffh)) / 2.0
        perf_var_norm = min(1.0, max(0.0, performance_variance * 100.0))
        aleatoric = 0.6 * perf_var_norm + 0.4 * min(1.0, osc * 1000.0)
        return float(aleatoric)


