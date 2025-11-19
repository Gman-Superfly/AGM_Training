import os
import json
import subprocess
from pathlib import Path


def test_hmpo_ingest_runner_smoke(tmp_path: Path):
    if os.environ.get("RUN_SMOKE") != "1":
        return  # opt-in only
    # Create a tiny JSONL with HMPO-like records
    logs = tmp_path / "hmpo.jsonl"
    with logs.open("w", encoding="utf-8") as f:
        for step in range(3):
            rec = {
                "step": step,
                "arithmetic_history": [1.0, 0.9, 0.8],
                "harmonic_history": [0.5, 0.55, 0.6],
                "td_mean": 0.0,
                "td_var": 0.01,
                "q_rel_change": 0.0,
                "eval_reward": float(step),
                "muon_clip_active": False,
                "muon_clip_rate": 0.0,
            }
            f.write(json.dumps(rec) + "\n")
    out = tmp_path / "out.jsonl"
    proc = subprocess.run(
        ["py", "-3", "scripts/hmpo_ingest_runner.py", "--logs", str(logs), "--format", "jsonl", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out.exists()
    assert out.stat().st_size > 0


