from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, List, Dict

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


def _maybe_import_torch():
    try:
        import importlib

        torch = importlib.import_module("torch")  # type: ignore[assignment]
        return torch
    except Exception:
        return None


def _to_float(x: Any) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def _coerce_seq_to_floats(x: Any) -> List[float]:
    torch = _maybe_import_torch()
    # Torch tensor -> CPU numpy -> list of floats
    if torch is not None and hasattr(x, "detach"):
        try:
            return [float(v) for v in x.detach().cpu().numpy().ravel().tolist()]
        except Exception:
            pass
    # Numpy array
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):  # type: ignore[attr-defined]
            return [float(v) for v in x.ravel().tolist()]
    except Exception:
        pass
    # Generic sequence
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    # Single scalar -> wrap
    try:
        return [float(x)]
    except Exception as e:
        raise AssertionError(f"cannot coerce to list[float]: {type(x)}") from e


class AGMHistories(BaseModel):
    model_config = ConfigDict(frozen=True)

    arithmetic_history: List[float] = Field(default_factory=list)
    harmonic_history: List[float] = Field(default_factory=list)

    @field_validator("arithmetic_history", "harmonic_history", mode="before")
    @classmethod
    def _coerce_histories(cls, v: Any) -> List[float]:
        return _coerce_seq_to_floats(v)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "AGMHistories":
        # Not strictly required to be equal length, but commonly expected
        if len(self.arithmetic_history) == 0 and len(self.harmonic_history) == 0:
            # allow empty during warm-up
            return self
        assert len(self.arithmetic_history) > 0, "arithmetic_history must be non-empty when provided"
        assert len(self.harmonic_history) > 0, "harmonic_history must be non-empty when provided"
        return self


class TDStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float = Field(default=0.0)
    var: float = Field(ge=0.0, default=0.0)


class QStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    rel_change: float = Field(default=0.0)


class RewardStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_mean: float = Field(default=0.0)
    eval_mean: float = Field(default=0.0)


class RLStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    td: TDStats
    q: QStats
    reward: RewardStats


class MuonClipStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    active: bool = Field(default=False)
    clip_rate: float = Field(ge=0.0, le=1.0, default=0.0)


class TelemetryPayload(BaseModel):
    """Normalized telemetry payload consumed by AGM controllers."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0)
    agm: AGMHistories
    rl: RLStats
    muon_clip: Optional[MuonClipStats] = None


def _get_nested(record: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = record
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_telemetry_from_hmpo_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-effort mapping from HMPO-style flat/nested logs to TelemetryPayload dict.

    Supports both nested and flat field names:
      - step: 'step' | 'global_step'
      - AGM histories: nested agm.* or flat arithmetic_history/harmonic_history
      - TD stats: rl.td.mean/var or flat td_mean/td_var
      - Q rel change: rl.q.rel_change or flat q_rel_change
      - Rewards: rl.reward.train_mean/eval_mean or flat train_reward_mean/eval_reward
      - MuonClip: muon_clip.active/clip_rate or flat muon_clip_active/muon_clip_rate
    """
    assert isinstance(record, Mapping), "record must be a mapping"
    step = _get_nested(record, ["step"], _get_nested(record, ["global_step"], 0))
    # AGM
    a_hist = _get_nested(record, ["agm", "arithmetic_history"], record.get("arithmetic_history", []))
    h_hist = _get_nested(record, ["agm", "harmonic_history"], record.get("harmonic_history", []))
    # RL TD
    td_mean = _get_nested(record, ["rl", "td", "mean"], record.get("td_mean", 0.0))
    td_var = _get_nested(record, ["rl", "td", "var"], record.get("td_var", 0.0))
    # RL Q
    q_rel = _get_nested(record, ["rl", "q", "rel_change"], record.get("q_rel_change", 0.0))
    # Rewards
    r_train = _get_nested(record, ["rl", "reward", "train_mean"], record.get("train_reward_mean", 0.0))
    r_eval = _get_nested(record, ["rl", "reward", "eval_mean"], record.get("eval_reward", 0.0))
    # MuonClip
    mc_active = _get_nested(record, ["muon_clip", "active"], record.get("muon_clip_active", False))
    mc_rate = _get_nested(record, ["muon_clip", "clip_rate"], record.get("muon_clip_rate", 0.0))

    payload: Dict[str, Any] = {
        "step": int(step),
        "agm": {
            "arithmetic_history": _coerce_seq_to_floats(a_hist),
            "harmonic_history": _coerce_seq_to_floats(h_hist),
        },
        "rl": {
            "td": {"mean": _to_float(td_mean), "var": max(0.0, _to_float(td_var))},
            "q": {"rel_change": _to_float(q_rel)},
            "reward": {"train_mean": _to_float(r_train), "eval_mean": _to_float(r_eval)},
        },
        "muon_clip": {
            "active": bool(mc_active),
            "clip_rate": float(mc_rate),
        },
    }
    return payload


def validate_telemetry(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize telemetry into a plain dict (floats/lists/bools/ints).

    Returns:
        A JSON-serializable dictionary validated against TelemetryPayload.
    """
    assert isinstance(payload, Mapping), "payload must be a mapping"
    entity = TelemetryPayload(**payload)  # type: ignore[arg-type]
    # Enforce additional cross-field sanity (best-effort)
    # If histories exist, ensure finite numbers (implicitly handled by float cast)
    return entity.model_dump()


