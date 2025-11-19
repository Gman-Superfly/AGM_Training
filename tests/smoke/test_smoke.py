import os
import subprocess


def test_learner_script_runs_smoke(tmp_path):
    if os.environ.get("RUN_SMOKE") != "1":
        return  # opt-in only
    # Run learner with default config; just ensure it starts and exits quickly
    proc = subprocess.run(["py", "-3", "scripts/learner.py", "--config", "configs/default.yaml"], capture_output=True, text=True)
    assert proc.returncode == 0

