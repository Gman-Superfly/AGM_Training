from __future__ import annotations

from typing import Mapping

from .config import AGMConfigEntity
from .entities import EarlyStopDecisionEntity, EarlyStopMetrics


class EarlyStopController:
    """Q-grounded, multi-criteria early stopping controller (minimal MVP)."""

    def __init__(self, config: AGMConfigEntity):
        assert config is not None, "config required"
        self.config = config
        self._patience_left = config.early_stopping.patience
        self._reward_hist: list[float] = []

    def update(
        self,
        td_error_stats: Mapping[str, float],
        q_stability: Mapping[str, float],
        eval_metrics: Mapping[str, float],
    ) -> EarlyStopDecisionEntity:
        assert td_error_stats is not None, "td_error_stats required"
        assert q_stability is not None, "q_stability required"
        assert eval_metrics is not None, "eval_metrics required"

        td_mean = float(td_error_stats.get("mean", 0.0))
        td_var = float(td_error_stats.get("var", 0.0))
        q_rel_change = float(q_stability.get("rel_change", 0.0))
        val_reward = float(eval_metrics.get("val_reward", 0.0))
        # Maintain reward history for optional trend test
        self._reward_hist.append(val_reward)
        if len(self._reward_hist) > max(2, int(self.config.early_stopping.trend_window)):
            self._reward_hist = self._reward_hist[-int(self.config.early_stopping.trend_window) :]

        # Simple heuristic: if improvement stalls and TD variance low, count down
        plateau_thresh = self.config.early_stopping.plateau_threshold
        hysteresis = 1.0 - self.config.early_stopping.hysteresis_pct
        improving = eval_metrics.get("improvement", 0.0) > plateau_thresh
        stable_q = abs(q_rel_change) < (plateau_thresh * hysteresis)
        low_td_var = td_var < (plateau_thresh * hysteresis)

        # Optional cost-based decision (weights sum not enforced; user controls)
        if self.config.early_stopping.use_cost:
            td_component = float(td_var)
            stability_component = float(abs(q_rel_change))
            plateau_component = float(max(0.0, plateau_thresh - eval_metrics.get("improvement", 0.0)))
            cost = (
                self.config.early_stopping.td_weight * td_component
                + self.config.early_stopping.stability_weight * stability_component
                + self.config.early_stopping.plateau_weight * plateau_component
            )
            # Count down when cost below hysteresis-weighted threshold
            if cost < plateau_thresh * hysteresis:
                self._patience_left = max(0, self._patience_left - 1)
            else:
                self._patience_left = self.config.early_stopping.patience
        else:
            # Optional trend test (slope <= threshold implies plateau)
            trend_ok = True
            if getattr(self.config.early_stopping, "enable_trend_test", False) and len(self._reward_hist) >= 2:
                n = len(self._reward_hist)
                sx = (n - 1) * n / 2.0
                sxx = (n - 1) * n * (2 * n - 1) / 6.0
                sy = sum(self._reward_hist)
                sxy = sum(i * v for i, v in enumerate(self._reward_hist))
                denom = n * sxx - sx * sx
                slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
                trend_ok = slope <= float(self.config.early_stopping.trend_slope_threshold)
            if (not improving) and stable_q and low_td_var and trend_ok:
                self._patience_left = max(0, self._patience_left - 1)
            else:
                # Reset patience if progress resumes
                self._patience_left = self.config.early_stopping.patience


        decision = EarlyStopDecisionEntity(
            should_stop=self._patience_left == 0,
            reason=(
                "plateau_low_variance" if self._patience_left == 0 else "continue"
            ),
            patience_left=self._patience_left,
            metrics=EarlyStopMetrics(
                td_mean=td_mean, td_var=td_var, q_rel_change=q_rel_change, val_reward=val_reward
            ),
        )
        return decision

