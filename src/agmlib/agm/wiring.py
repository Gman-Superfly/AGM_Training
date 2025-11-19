from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .telemetry import validate_telemetry
from .convergence import AdaptiveAGMController
from .phase import TrainingPhaseAdaptiveAGM
from .hparam import AGMHyperparameterAdapter, BaseHyperparams, HParamAdapterConfig
from .multiscale import MultiScaleAGMFramework
from .detectors import AGMConvergenceDetector, AGMUncertaintyEstimator


class ControllerSuite:
    """Stateful wrapper composing AGM controllers for easy integration in training loops.

    Usage:
        suite = ControllerSuite()
        decisions = suite.update(telemetry_payload_dict)
        # Apply decisions['hyperparams'] (bounded) at your discretion.
    """

    def __init__(
        self,
        *,
        base_hyperparams: Mapping[str, Any] | None = None,
        hparam_config: HParamAdapterConfig | None = None,
        enable_phase: bool = True,
        enable_hparams: bool = True,
        dwell_steps: int = 5,
        divergence_slope_threshold: float = -0.05,
    ) -> None:
        self.conv = AdaptiveAGMController()
        self.phase = TrainingPhaseAdaptiveAGM()
        base = base_hyperparams or BaseHyperparams().model_dump()
        self.hparams = AGMHyperparameterAdapter(base_hyperparams=base, config=hparam_config)
        self.multiscale = MultiScaleAGMFramework()
        self.detector = AGMConvergenceDetector()
        self.uncertainty = AGMUncertaintyEstimator()
        # Toggles and guardrails
        self.enable_phase = bool(enable_phase)
        self.enable_hparams = bool(enable_hparams)
        self.dwell_steps = int(max(0, dwell_steps))
        self.divergence_slope_threshold = float(divergence_slope_threshold)
        self._last_change_step: int = -10**9
        self._last_hparams: Dict[str, float] | None = None
        self._rate_hist: list[float] = []

    def update(self, telemetry: Mapping[str, Any]) -> Dict[str, Any]:
        tel = validate_telemetry(telemetry)
        step = int(tel["step"])
        a_hist = tel["agm"]["arithmetic_history"]
        h_hist = tel["agm"]["harmonic_history"]
        if len(a_hist) > 0 and len(h_hist) > 0:
            rate = self.conv.update(a_hist, h_hist)
            spread = abs(a_hist[-1] - h_hist[-1]) / (abs(a_hist[-1]) + abs(h_hist[-1]) + 1e-8)
        else:
            rate = 0.5
            spread = 0.0
        # Track rate history for divergence detection
        self._rate_hist.append(float(rate))
        if len(self._rate_hist) > 10:
            self._rate_hist = self._rate_hist[-10:]
        agm_metrics = {
            "convergence_rate": rate,
            "mean_spread": spread,
            "iterations_used": max(len(a_hist), len(h_hist)),
            "timestamp": step,
        }
        self.multiscale.update_all_trackers(agm_metrics)
        hierarchical = self.multiscale.compute_hierarchical_adaptation()
        # Phase (toggle)
        if self.enable_phase:
            strategy, strategy_cfg = self.phase.select_adaptive_strategy(
                loss_history=[],
                performance_metrics=tel.get("rl", {}),
                agm_convergence_rate=rate,
            )
        else:
            strategy, strategy_cfg = "disabled", {}
        # Hyperparams (toggle with dwell and rollback guardrails)
        dwell_blocked = False
        rolled_back = False
        if self.enable_hparams:
            # Divergence detection via simple slope over last window
            slope = 0.0
            n = len(self._rate_hist)
            if n >= 3:
                sx = (n - 1) * n / 2.0
                sxx = (n - 1) * n * (2 * n - 1) / 6.0
                sy = sum(self._rate_hist)
                sxy = sum(i * v for i, v in enumerate(self._rate_hist))
                denom = n * sxx - sx * sx
                slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
            # Compute candidate suggestions
            cand = self.hparams.adapt_hyperparameters(agm_state={"agm": tel["agm"]}, performance_trend=0.0)
            # Apply dwell-time limit
            if step - self._last_change_step < self.dwell_steps and self._last_hparams is not None:
                dwell_blocked = True
                suggestions = dict(self._last_hparams)
            # Rollback on divergence immediately after a change
            elif slope <= self.divergence_slope_threshold and self._last_hparams is not None:
                rolled_back = True
                suggestions = dict(self._last_hparams)
            else:
                suggestions = cand
                self._last_hparams = dict(cand)
                self._last_change_step = step
        else:
            suggestions = {"learning_rate": BaseHyperparams().learning_rate, "epsilon": BaseHyperparams().epsilon, "batch_size": float(BaseHyperparams().batch_size), "agm_iterations": float(BaseHyperparams().agm_iterations)}
        should_stop, reason = self.detector.should_stop_training(
            agm_metrics=agm_metrics,
            performance_metrics={"primary_metric": float(tel["rl"]["reward"]["eval_mean"])},
        )
        epi = self.uncertainty.estimate_epistemic_uncertainty(tel["agm"])
        alea = self.uncertainty.estimate_aleatoric_uncertainty(tel["agm"], performance_variance=0.0)
        return {
            "step": step,
            "convergence_rate": rate,
            "phase": strategy,
            "phase_config": strategy_cfg,
            "hyperparams": suggestions,
            "hierarchical": hierarchical,
            "early_stop": {"should_stop": should_stop, "reason": reason},
            "uncertainty": {"epistemic": epi, "aleatoric": alea},
            "toggles": {"phase": self.enable_phase, "hparams": self.enable_hparams},
            "guardrails": {"dwell_blocked": dwell_blocked, "rolled_back": rolled_back, "dwell_steps": self.dwell_steps},
        }


