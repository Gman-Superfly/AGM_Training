# HMPO Integration Guide (Telemetry → Controllers → Decisions)

This guide shows how to connect HMPO telemetry to the AGM controllers in this repo, both offline (ingestion runner) and live (wiring helper).

## Telemetry schema
- Controllers consume a dict validated by `agmlib.agm.telemetry.TelemetryPayload`:
  - `step: int`
  - `agm.{arithmetic_history,harmonic_history}: list[float]`
  - `rl.td.{mean,var}: float`, `rl.q.rel_change: float`, `rl.reward.{train_mean,eval_mean}: float`
  - `muon_clip.{active,clip_rate}: bool,float`
- Use `build_telemetry_from_hmpo_record(record)` to map HMPO JSONL/CSV records (nested or flat) into this schema.

## Offline (Windows PowerShell) — ingestion runner
```powershell
# Process HMPO logs and write controller decisions
uv run python scripts\hmpo_ingest_runner.py --logs .\hmpo\metrics.jsonl --format jsonl --out .\logs\controllers.jsonl

# Quick summary
uv run python scripts\hmpo_ingest_runner.py --logs .\hmpo\metrics.jsonl --format jsonl --out .\logs\controllers.jsonl --summary --summary-out .\logs\controllers_summary.json

# Optional tuning via config files
uv run python scripts\hmpo_ingest_runner.py --logs .\hmpo\metrics.jsonl --format jsonl --out .\logs\controllers.jsonl --hparam-config .\configs\agm_hparam.yaml --phase-config .\configs\agm_phase.yaml --convergence-config .\configs\agm_convergence.yaml

# Analyze results (optionally include HMPO logs to compute time-to-target reward)
uv run python scripts\analyze_ingest.py --controllers .\logs\controllers.jsonl --hmpo .\hmpo\metrics.jsonl --target-reward 200
Get-Content .\logs\experiments\* \summary.json
```

## Live — wiring in HMPO loop
Minimal example using the `ControllerSuite` for dict-in/dict-out decisions:
```python
from agmlib.agm import ControllerSuite, build_telemetry_from_hmpo_record, AGMCurriculumController, CurriculumConfig

suite = ControllerSuite()
curriculum = AGMCurriculumController(CurriculumConfig())

# At eval/interval in HMPO:
record = {
    "step": global_step,
    # Provide AGM histories and RL stats from HMPO logs/state
    "arithmetic_history": arithmetic_hist,  # list[float]
    "harmonic_history": harmonic_hist,      # list[float]
    "td_mean": td_mean,
    "td_var": td_var,
    "q_rel_change": q_rel_change,
    "eval_reward": eval_reward,
    "muon_clip_active": muon_clip_active,
    "muon_clip_rate": muon_clip_rate,
}
tel = build_telemetry_from_hmpo_record(record)
decisions = suite.update(tel)

# Apply bounded suggestions (example: LR/epsilon)
hp = decisions["hyperparams"]
optimizer.lr = float(hp["learning_rate"])
epsilon = float(hp["epsilon"])

# Optional: handle early stop
if decisions["early_stop"]["should_stop"]:
    reason = decisions["early_stop"]["reason"]
    # checkpoint/save and exit loop

# Optional: curriculum update (requires success_rate)
action = curriculum.update_curriculum(
    agm_convergence_rate=float(decisions["convergence_rate"]),
    success_rate=float(eval_success_rate),  # supply from HMPO evaluation
    performance_metrics={"eval_reward": float(eval_reward)},
)
if action == "increase_difficulty":
    # advance to next curriculum level in HMPO
    pass
```

## Early-stop trend enhancement
- Config is in `AGMConfigEntity.early_stopping`:
  - `enable_trend_test: bool` (default False)
  - `trend_window: int` (default 10)
  - `trend_slope_threshold: float` (default 0.0)
- When enabled, the controller counts down patience if reward slope over `trend_window` is ≤ `trend_slope_threshold` and other stability checks pass.

## Uncertainty gating (offline ingestion)
- Use runner flags to enable gating of hyperparameter suggestions on high uncertainty:
  - `--uncertainty-enabled`
  - `--uncertainty-mode total|epistemic|aleatoric`
  - `--uncertainty-threshold 0.7`


## CI (optional)
- GitHub Actions Windows workflow `.github/workflows/windows-ci.yml` runs unit tests on pushes/PRs.


