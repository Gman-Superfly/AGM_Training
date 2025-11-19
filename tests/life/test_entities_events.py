from pathlib import Path
import json

from agmlib.entities import SmoothingDecisionEntity
from agmlib.events import AdaptationAppliedEvent, emit_event


def test_event_emission(tmp_path: Path) -> None:
    decision = SmoothingDecisionEntity(
        applied=True, alpha_step_used=0.1, tau_clip=0.1, gated_by_uncertainty=True
    )
    sink = tmp_path / "events.jsonl"
    emit_event(
        AdaptationAppliedEvent(
            subject_id=str(decision.ecs_id),
            step=1,
            decision={"alpha_step": 0.1},
            pre_metrics={"m": 0},
            post_metrics={"m": 1},
        ),
        sink_path=sink,
    )
    data = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    assert any(e.get("name") == "AdaptationAppliedEvent" for e in data)

