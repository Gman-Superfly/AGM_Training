from agmlib.agm.telemetry import build_telemetry_from_hmpo_record, validate_telemetry


def test_flat_record_mapping_and_validation():
    rec = {
        "step": 123,
        "arithmetic_history": [1.0, 0.9, 0.8],
        "harmonic_history": [0.5, 0.55, 0.6],
        "td_mean": 0.01,
        "td_var": 0.02,
        "q_rel_change": 0.001,
        "train_reward_mean": 1.5,
        "eval_reward": 1.2,
        "muon_clip_active": False,
        "muon_clip_rate": 0.0,
    }
    tel = build_telemetry_from_hmpo_record(rec)
    norm = validate_telemetry(tel)
    assert norm["step"] == 123
    assert len(norm["agm"]["arithmetic_history"]) == 3
    assert "td" in norm["rl"] and "q" in norm["rl"] and "reward" in norm["rl"]


def test_nested_record_mapping_and_validation():
    rec = {
        "global_step": 7,
        "agm": {
            "arithmetic_history": [0.2, 0.21],
            "harmonic_history": [0.1, 0.11],
        },
        "rl": {
            "td": {"mean": 0.0, "var": 0.0},
            "q": {"rel_change": 0.0},
            "reward": {"train_mean": 0.0, "eval_mean": 0.0},
        },
        "muon_clip": {"active": True, "clip_rate": 0.25},
    }
    tel = build_telemetry_from_hmpo_record(rec)
    norm = validate_telemetry(tel)
    assert norm["step"] == 7
    assert norm["muon_clip"]["active"] is True
    assert 0.0 <= norm["muon_clip"]["clip_rate"] <= 1.0


