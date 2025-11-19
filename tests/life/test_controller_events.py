from pathlib import Path
import json

from agmlib.events import AdaptationAppliedEvent, emit_event


def test_controller_decision_event_emitted(tmp_path: Path) -> None:
    sink = tmp_path / "events.jsonl"
    decision = {"phase": "standard_agm", "hparams": {"learning_rate": 1e-3}}
    emit_event(
        AdaptationAppliedEvent(
            step=42,
            decision=decision,
            pre_metrics={"conv_rate": 0.6},
            post_metrics={"hierarchical": {"immediate": "continue_current"}},
        ),
        sink_path=sink,
    )
    data = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    assert any(e.get("name") == "AdaptationAppliedEvent" for e in data)


