from __future__ import annotations

from typing import Any, List, Sequence, Optional

from pydantic import BaseModel, Field, ConfigDict


def _maybe_import_np():
    try:
        import numpy as _np  # type: ignore

        return _np
    except Exception:
        return None


def _maybe_import_torch():
    try:
        import importlib

        torch = importlib.import_module("torch")  # type: ignore[assignment]
        return torch
    except Exception:
        return None


def _to_1d_array(x: Any) -> "list[float] | Any":
    """Return a 1D float list for generic sequences or tensors; otherwise best-effort."""
    torch = _maybe_import_torch()
    if torch is not None and hasattr(x, "detach"):
        try:
            return [float(v) for v in x.detach().cpu().numpy().ravel().tolist()]
        except Exception:
            pass
    np = _maybe_import_np()
    if np is not None:
        try:
            if isinstance(x, np.ndarray):  # type: ignore[attr-defined]
                return [float(v) for v in x.ravel().tolist()]
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    # Fallback: scalar -> single-element list
    try:
        return [float(x)]
    except Exception as e:
        raise AssertionError(f"cannot coerce to 1D float array: {type(x)}") from e


class ConvergenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    slow_threshold: float = Field(ge=0.0, le=1.0, default=0.2)
    unstable_variance_threshold: float = Field(ge=0.0, default=0.1)
    regime_window: int = Field(ge=1, default=10)
    history_cap: int = Field(ge=10, default=1000)


class AdaptiveAGMController:
    """Monitor AGM convergence and classify training regimes.

    - Convergence rate: 1 - mean(|A - H| / (A + H + eps))
    - Keeps a bounded history of recent rates for regime detection
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None) -> None:
        self.config = config or ConvergenceConfig()
        self._history: List[float] = []

    @property
    def convergence_history(self) -> List[float]:
        return list(self._history)

    def compute_convergence_rate(self, arithmetic: Sequence[float] | Any, harmonic: Sequence[float] | Any) -> float:
        a = _to_1d_array(arithmetic)
        h = _to_1d_array(harmonic)
        n = min(len(a), len(h))
        assert n >= 1, "histories must contain at least one element"
        eps = 1e-8
        total = 0.0
        for i in range(n):
            denom = abs(a[i]) + abs(h[i]) + eps
            total += abs(a[i] - h[i]) / denom
        relative_gap = total / float(n)
        rate = 1.0 - relative_gap
        # Bound to [0, 1]
        if rate < 0.0:
            rate = 0.0
        if rate > 1.0:
            rate = 1.0
        return float(rate)

    def update(self, arithmetic: Sequence[float] | Any, harmonic: Sequence[float] | Any) -> float:
        """Update controller with the latest AGM series and return current rate."""
        rate = self.compute_convergence_rate(arithmetic, harmonic)
        self._history.append(rate)
        # Cap history to avoid unbounded growth
        if len(self._history) > self.config.history_cap:
            self._history = self._history[-self.config.history_cap :]
        return rate

    def detect_training_regime(self) -> str:
        """Classify training regime from recent convergence history."""
        if len(self._history) < self.config.regime_window:
            return "initialization"
        window = self._history[-self.config.regime_window :]
        # Mean and variance over the window
        m = sum(window) / float(len(window))
        v = 0.0
        for r in window:
            v += (r - m) * (r - m)
        v /= float(len(window))
        if m > self.config.fast_threshold:
            return "stable_learning"
        if m < self.config.slow_threshold:
            return "struggling"
        if v > self.config.unstable_variance_threshold:
            return "unstable"
        return "normal_learning"


