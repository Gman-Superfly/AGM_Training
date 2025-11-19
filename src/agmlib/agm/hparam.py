from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Optional, Tuple, List

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


class BaseHyperparams(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(gt=0.0, default=1e-3)
    epsilon: float = Field(ge=0.0, le=1.0, default=0.1)
    batch_size: int = Field(ge=1, default=256)
    agm_iterations: int = Field(ge=1, default=5)


class HParamBounds(BaseModel):
    model_config = ConfigDict(frozen=True)

    lr_min: float = Field(gt=0.0, default=1e-5)
    lr_max: float = Field(gt=0.0, default=5e-3)
    eps_min: float = Field(ge=0.0, le=1.0, default=0.01)
    eps_max: float = Field(ge=0.0, le=1.0, default=0.5)
    batch_min: int = Field(ge=1, default=64)
    batch_max: int = Field(ge=1, default=2048)
    agm_min: int = Field(ge=1, default=1)
    agm_max: int = Field(ge=1, default=20)


class HParamAdapterConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    smoothing_factor: float = Field(ge=0.0, le=1.0, default=0.9)
    bounds: HParamBounds = Field(default_factory=HParamBounds)
    # Multipliers
    lr_fast_mult: float = Field(ge=0.0, default=1.3)
    lr_slow_mult: float = Field(ge=0.0, default=0.7)
    lr_unstable_mult: float = Field(ge=0.0, default=0.6)
    eps_spread_high_mult: float = Field(ge=0.0, default=1.4)
    eps_spread_low_mult: float = Field(ge=0.0, default=0.8)
    batch_stable_big_mult: float = Field(ge=0.0, default=1.5)
    batch_unstable_small_mult: float = Field(ge=0.0, default=0.7)
    # Thresholds
    fast_conv_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    slow_conv_threshold: float = Field(ge=0.0, le=1.0, default=0.3)
    stability_high: float = Field(ge=0.0, le=1.0, default=0.8)
    stability_low: float = Field(ge=0.0, le=1.0, default=0.3)
    spread_high: float = Field(ge=0.0, default=0.3)
    spread_low: float = Field(ge=0.0, default=0.05)


def _convergence_speed(agm_state: Mapping[str, Any]) -> float:
    agm = agm_state.get("agm", agm_state)
    a_hist = _to_float_list(agm.get("arithmetic_history", []))
    h_hist = _to_float_list(agm.get("harmonic_history", []))
    n = min(len(a_hist), len(h_hist))
    if n < 2:
        return 0.5
    gaps = [abs(a_hist[i] - h_hist[i]) for i in range(n)]
    if gaps[0] <= 1e-8:
        return 0.5
    gap_reduction = max(0.0, (gaps[0] - gaps[-1]) / gaps[0])
    return min(1.0, float(gap_reduction))


def _mean_spread(agm_state: Mapping[str, Any]) -> float:
    agm = agm_state.get("agm", agm_state)
    a_hist = _to_float_list(agm.get("arithmetic_history", []))
    h_hist = _to_float_list(agm.get("harmonic_history", []))
    if not a_hist or not h_hist:
        return 0.0
    a = a_hist[-1]
    h = h_hist[-1]
    denom = abs(a) + abs(h) + 1e-8
    return float(abs(a - h) / denom)


def _stability(agm_state: Mapping[str, Any]) -> float:
    agm = agm_state.get("agm", agm_state)
    a_hist = _to_float_list(agm.get("arithmetic_history", []))[-10:]
    h_hist = _to_float_list(agm.get("harmonic_history", []))[-10:]
    if len(a_hist) < 5 or len(h_hist) < 5:
        return 0.5
    # Variance over recent windows
    def _var(values: Sequence[float]) -> float:
        m = sum(values) / float(len(values))
        return sum((v - m) * (v - m) for v in values) / float(len(values))

    combined = _var(a_hist) + _var(h_hist)
    return float(1.0 / (1.0 + combined * 100.0))


class AGMHyperparameterAdapter:
    """Suggest hyperparameter adjustments from AGM convergence properties."""

    def __init__(self, base_hyperparams: Mapping[str, Any] | BaseHyperparams, config: HParamAdapterConfig | None = None):
        self.base = base_hyperparams if isinstance(base_hyperparams, BaseHyperparams) else BaseHyperparams(**base_hyperparams)
        self.cfg = config or HParamAdapterConfig()
        self._prev: Dict[str, float] | None = None

    def adapt_hyperparameters(self, *, agm_state: Mapping[str, Any], performance_trend: float) -> Dict[str, float]:
        c = self.cfg
        bounds = c.bounds
        conv = _convergence_speed(agm_state)
        spread = _mean_spread(agm_state)
        stab = _stability(agm_state)

        # Learning rate
        lr = float(self.base.learning_rate)
        if conv > c.fast_conv_threshold:
            lr *= c.lr_fast_mult
        elif conv < c.slow_conv_threshold:
            lr *= c.lr_slow_mult
        elif stab < c.stability_low:
            lr *= c.lr_unstable_mult
        lr = min(bounds.lr_max, max(bounds.lr_min, lr))

        # Epsilon
        eps = float(self.base.epsilon)
        if spread > c.spread_high:
            eps *= c.eps_spread_high_mult
        elif spread < c.spread_low:
            eps *= c.eps_spread_low_mult
        eps = min(bounds.eps_max, max(bounds.eps_min, eps))

        # Batch size
        bs = int(self.base.batch_size)
        if stab > c.stability_high and conv > c.fast_conv_threshold:
            bs = int(min(bounds.batch_max, max(bounds.batch_min, int(bs * c.batch_stable_big_mult))))
        elif stab < c.stability_low:
            bs = int(min(bounds.batch_max, max(bounds.batch_min, int(bs * c.batch_unstable_small_mult))))

        # AGM iterations
        if conv < c.slow_conv_threshold:
            agm_iters = 10
        elif conv > c.fast_conv_threshold:
            agm_iters = 3
        else:
            agm_iters = 5
        agm_iters = int(min(bounds.agm_max, max(bounds.agm_min, agm_iters)))

        suggestions: Dict[str, float] = {
            "learning_rate": float(lr),
            "epsilon": float(eps),
            "batch_size": float(bs),
            "agm_iterations": float(agm_iters),
        }

        # Smooth to prevent oscillations
        if self._prev is not None and c.smoothing_factor > 0.0:
            alpha = c.smoothing_factor
            for k in list(suggestions.keys()):
                suggestions[k] = float(alpha * self._prev.get(k, suggestions[k]) + (1.0 - alpha) * suggestions[k])
        self._prev = suggestions.copy()
        return suggestions


