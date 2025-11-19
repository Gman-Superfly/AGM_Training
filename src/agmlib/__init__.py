"""AGM Training library scaffold.

Entity-first, event-driven, conflict-aware components for adaptive RL training.
"""

from .config import AGMConfigEntity, load_config
from .entities import (
    AGMMetricsEntity,
    SmoothingDecisionEntity,
    EarlyStopDecisionEntity,
    PopulationMetricsEntity,
)
from .events import (
    BaseEvent,
    AdaptationAppliedEvent,
    EarlyStopTriggeredEvent,
    ParameterSyncEvent,
    ReplayUpdateEvent,
    emit_event,
)
from .smoothing import (
    schedule_alpha_step,
    compute_knn,
    adaptive_sigmas,
    kernel_consensus,
    trust_region_update,
    featurewise_scale_preservation,
    average_disagreement,
    should_apply_smoothing_decision,
    apply_batch_smoothing,
    free_energy,
)
from .adaptation import AdaptiveSmoothingController, MultiScaleEma
from .agm.convergence import AdaptiveAGMController
from .agm.phase import TrainingPhaseDetector, TrainingPhaseAdaptiveAGM
from .agm.hparam import AGMHyperparameterAdapter
from .agm.multiscale import MultiScaleAGMFramework
from .agm.detectors import AGMConvergenceDetector, AGMUncertaintyEstimator
from .agm.telemetry import validate_telemetry, build_telemetry_from_hmpo_record, TelemetryPayload

__all__ = [
    "AGMConfigEntity",
    "load_config",
    "AGMMetricsEntity",
    "SmoothingDecisionEntity",
    "EarlyStopDecisionEntity",
    "PopulationMetricsEntity",
    "BaseEvent",
    "AdaptationAppliedEvent",
    "EarlyStopTriggeredEvent",
    "ParameterSyncEvent",
    "ReplayUpdateEvent",
    "emit_event",
    "schedule_alpha_step",
    "compute_knn",
    "adaptive_sigmas",
    "kernel_consensus",
    "trust_region_update",
    "featurewise_scale_preservation",
    "average_disagreement",
    "should_apply_smoothing_decision",
    "apply_batch_smoothing",
    "free_energy",
    "AdaptiveSmoothingController",
    "MultiScaleEma",
    "AdaptiveAGMController",
    "TrainingPhaseDetector",
    "TrainingPhaseAdaptiveAGM",
    "AGMHyperparameterAdapter",
    "MultiScaleAGMFramework",
    "AGMConvergenceDetector",
    "AGMUncertaintyEstimator",
    "validate_telemetry",
    "build_telemetry_from_hmpo_record",
    "TelemetryPayload",
]
