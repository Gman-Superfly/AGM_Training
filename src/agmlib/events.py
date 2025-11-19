from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    subject_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    process_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__


class AdaptationAppliedEvent(BaseEvent):
    step: int
    decision: dict
    pre_metrics: dict
    post_metrics: dict


class EarlyStopTriggeredEvent(BaseEvent):
    step: int
    decision: dict


class ParameterSyncEvent(BaseEvent):
    learner_id: str
    step: int
    checksum: str
    model_version: str


class ReplayUpdateEvent(BaseEvent):
    shard_id: str
    count: int
    priority_stats: dict


def _to_serializable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, BaseModel):
        return json.loads(obj.model_dump_json())
    if hasattr(obj, "__dict__"):
        try:
            return obj.__dict__
        except Exception:
            return str(obj)
    return obj


def emit_event(event: BaseEvent, *, sink_path: Optional[Path] = None) -> None:
    """Emit event to JSONL sink (logs/events.jsonl) and stdout.

    Ensures directory exists; appends a single JSON object per line.
    """
    assert event is not None, "event required"
    sink = sink_path or Path("logs") / "events.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": event.name,
        **json.loads(event.model_dump_json()),
    }
    line = json.dumps(payload, default=_to_serializable)
    with open(sink, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(line)

