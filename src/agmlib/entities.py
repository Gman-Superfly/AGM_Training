from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class BaseEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    ecs_id: UUID = Field(default_factory=uuid4)
    version: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AGMMetricsEntity(BaseEntity):
    convergence_rate: float = Field(ge=0.0, le=1.0)
    mean_spread: float = Field(ge=0.0)
    oscillation_strength: float = Field(ge=0.0)
    iterations_used: int = Field(ge=0)
    timestamp: int = Field(ge=0)


class EarlyStopMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    td_mean: float
    td_var: float
    q_rel_change: float
    val_reward: float


class SmoothingDecisionEntity(BaseEntity):
    applied: bool
    alpha_step_used: float = Field(ge=0.0)
    tau_clip: float = Field(ge=0.0)
    gated_by_uncertainty: bool
    fep_value: Optional[float] = None
    sigma_stats: Dict[str, float] = Field(default_factory=dict)


class EarlyStopDecisionEntity(BaseEntity):
    should_stop: bool
    reason: str
    patience_left: int = Field(ge=0)
    metrics: EarlyStopMetrics


class PopulationMetricsEntity(BaseEntity):
    diversity: float = Field(ge=0.0)
    strategy_entropy: float = Field(ge=0.0)

