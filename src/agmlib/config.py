from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field, ConfigDict
import yaml


class ReplayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(ge=1, default=256)
    prioritized_replay: bool = Field(default=True)
    alpha: float = Field(ge=0.0, le=1.0, default=0.6)
    beta0: float = Field(ge=0.0, le=1.0, default=0.4)


class KernelSmoothingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    k: int = Field(ge=1, default=16)
    knn_block_size: int | None = Field(default=None, description="Optional row-block size for kNN to control memory")
    sigma_min: float = Field(gt=0.0, default=1e-3)
    sigma_max: float = Field(gt=0.0, default=1.0)
    alpha_step_init: float = Field(gt=0.0, le=1.0, default=0.15)
    alpha_step_schedule: Literal["cosine", "constant"] = Field(default="cosine")
    alpha_step_decay_steps: int = Field(ge=1, default=100000)
    dim_exponent: float = Field(ge=0.0, le=2.0, default=0.5)
    trust_region_tau: float = Field(gt=0.0, default=0.1)
    latent_dim: int = Field(ge=1, default=64)
    gate_by_td_variance: bool = Field(default=True)
    td_var_min: float = Field(ge=0.0, default=0.0)
    scale_preservation: bool = Field(default=False, description="Enable feature-wise scale preservation after update")
    disagreement_min: float = Field(ge=0.0, default=0.0, description="Min average ||G-h|| to apply smoothing")
    noise_std: float = Field(ge=0.0, default=0.0, description="Stddev of Gaussian exploration noise η added to Δh before clipping")
    # Adaptive controller parameters
    adapt_enabled: bool = Field(default=True)
    adapt_rate: float = Field(ge=0.0, le=1.0, default=0.05)
    alpha_step_min: float = Field(ge=0.0, default=0.01)
    alpha_step_max: float = Field(ge=0.0, default=0.5)
    tau_min: float = Field(ge=0.0, default=0.01)
    tau_max: float = Field(ge=0.0, default=1.0)
    k_min: int = Field(ge=1, default=4)
    k_max: int = Field(ge=1, default=128)


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval_interval: int = Field(ge=1, default=5000)
    patience: int = Field(ge=1, default=10)
    hysteresis_pct: float = Field(ge=0.0, le=1.0, default=0.03)
    td_ema_beta: float = Field(ge=0.0, lt=1.0, default=0.9)
    plateau_threshold: float = Field(ge=0.0, default=0.5)
    use_cost: bool = Field(default=True, description="Use weighted multi-criteria cost for early stopping decision")
    td_weight: float = Field(ge=0.0, le=1.0, default=0.3)
    stability_weight: float = Field(ge=0.0, le=1.0, default=0.3)
    plateau_weight: float = Field(ge=0.0, le=1.0, default=0.4)
    # Optional trend enhancement
    enable_trend_test: bool = Field(default=False, description="Enable trend-based early-stop enhancement")
    trend_window: int = Field(ge=2, default=10)
    trend_slope_threshold: float = Field(default=0.0, description="Max slope to consider plateau (<= threshold)")


class DistributedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    actors_per_gpu: int = Field(ge=0, default=4)
    learners: int = Field(ge=1, default=1)
    param_sync_interval: int = Field(ge=1, default=1000)
    replay_shards: int = Field(ge=1, default=4)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(gt=0.0, default=1e-3)
    lr_min: float = Field(gt=0.0, default=1e-5)
    lr_max: float = Field(gt=0.0, default=5e-3)
    lr_adapt_rate: float = Field(ge=0.0, le=1.0, default=0.05)


class AGMConfigEntity(BaseModel):
    """Typed, immutable configuration entity for AGM training.

    Enforces disambiguation between alpha_step, dim_exponent, and replay.alpha.
    """

    model_config = ConfigDict(frozen=True)

    version: int = Field(ge=0, default=1)
    replay: ReplayConfig
    kernel_smoothing: KernelSmoothingConfig
    early_stopping: EarlyStoppingConfig
    distributed: DistributedConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def validate_config(self) -> None:
        # Core validation and cross-field checks
        assert self.kernel_smoothing.sigma_min <= self.kernel_smoothing.sigma_max, (
            f"sigma_min {self.kernel_smoothing.sigma_min} must be <= sigma_max {self.kernel_smoothing.sigma_max}"
        )
        assert self.kernel_smoothing.k < self.replay.batch_size, (
            "k must be strictly less than batch_size to ensure non-empty neighborhoods"
        )


def load_config(path: Union[str, Path]) -> AGMConfigEntity:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Immutable, validated `AGMConfigEntity`.
    """
    assert path is not None, "path required"
    path_obj = Path(path)
    assert path_obj.exists(), f"Config file not found: {path_obj}"
    with path_obj.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entity = AGMConfigEntity(
        replay=ReplayConfig(**raw.get("replay", {})),
        kernel_smoothing=KernelSmoothingConfig(**raw.get("kernel_smoothing", {})),
        early_stopping=EarlyStoppingConfig(**raw.get("early_stopping", {})),
        distributed=DistributedConfig(**raw.get("distributed", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    entity.validate_config()
    return entity

