## Developer Guide (Short)

This guide orients you to the scaffold and shows how to extend it safely under the Datamutant rules (assert early/often, type everything, entity-first, events, immutability).

### Setup
- Use uv and Python 3.10
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create venv and install:
  - `uv venv --python 3.10`
  - `uv pip install -r requirements.txt`
  - Optional Torch: `uv pip install torch torchvision torchaudio [...]`

### Project layout
- `src/agmlib/`
  - `config.py`: `AGMConfigEntity` (immutable, validated)
  - `entities.py`: `AGMMetricsEntity`, `SmoothingDecisionEntity`, `EarlyStopDecisionEntity`, etc.
  - `events.py`: event models + `emit_event` (JSONL sink)
  - `smoothing.py`: kNN, sigmas, kernel consensus, trust-region, gating, scheduler, `apply_batch_smoothing`
  - `early_stop.py`: `EarlyStopController` (MVP)
  - `telemetry.py`: `MetricsLogger`, `histogram`
- `src/agmlib/agm/`
  - `telemetry.py`: Telemetry validation and HMPO mapping (`TelemetryPayload`, `validate_telemetry`, `build_telemetry_from_hmpo_record`)
  - `convergence.py`: `AdaptiveAGMController`
  - `phase.py`: `TrainingPhaseDetector`, `TrainingPhaseAdaptiveAGM`
  - `hparam.py`: `AGMHyperparameterAdapter` (dict-in/out; clamps + smoothing)
  - `multiscale.py`: `MultiScaleAGMFramework`
  - `detectors.py`: `AGMConvergenceDetector`, `AGMUncertaintyEstimator`
- `scripts/`: `learner.py` (wiring/demo), `actor.py` (stub)
- `scripts/`: `hmpo_ingest_runner.py` (HMPO -> controllers streaming runner)
- `configs/default.yaml`: configuration defaults

### Core APIs you will use
- Config
  - Load: `from agmlib.config import load_config; cfg = load_config("configs/default.yaml")`
  - Fields (subset):
    - `kernel_smoothing.{k,sigma_min,sigma_max,alpha_step_init,alpha_step_schedule,alpha_step_decay_steps,dim_exponent,trust_region_tau,td_var_min,disagreement_min,scale_preservation,latent_dim}`
- Events
  - Emit: `from agmlib.events import emit_event, AdaptationAppliedEvent`
  - Sink: `logs/events.jsonl` (one JSON per line)
- Smoothing
  - High-level:
    - `from agmlib.smoothing import apply_batch_smoothing`
    - `h_updated, diag = apply_batch_smoothing(h=h, z=z, step=step, td_var=td_var, config=cfg)`
      - `diag` includes `gate`, `avg_disagreement`, `sigma_stats`, `delta_h_stats`, plus tensors for histograms
  - Low-level pieces (when needed): `compute_knn`, `adaptive_sigmas`, `kernel_consensus`, `trust_region_update`, `schedule_alpha_step`, `average_disagreement`, `should_apply_smoothing_decision`
- Telemetry
  - `from agmlib.telemetry import MetricsLogger, histogram`
  - Write JSONL: `MetricsLogger().log({"step": step, ...})` → `logs/metrics.jsonl`
- Controllers & ingestion
  - Import: `from agmlib.agm import AdaptiveAGMController, TrainingPhaseAdaptiveAGM, AGMHyperparameterAdapter, MultiScaleAGMFramework, AGMConvergenceDetector, AGMUncertaintyEstimator`
  - Validate telemetry: `from agmlib.agm import validate_telemetry, build_telemetry_from_hmpo_record`
  - Runner (PowerShell): `uv run python scripts\\hmpo_ingest_runner.py --logs .\\hmpo\\metrics.jsonl --format jsonl --out .\\logs\\controllers.jsonl`

### Roadmap & Phase Sources
- Primary roadmap with phases: `AGM_TRAINING.md` (Implementation Roadmap)
- AGM properties phase TODOs: `AGM_ADAPTIVE_PROPERTIES.md` (Implementation TODO Roadmap)
- Execution order/tasking: `ACTION_PLAN.md` (Timeline and Order of Execution)
- Completed work per phase: `CHANGELOG.md`

### Extending gating policy (example)
1) Add new thresholds to `KernelSmoothingConfig` in `config.py` (with types, ranges).
2) Use them inside `should_apply_smoothing_decision(...)` with explicit AND/OR logic.
3) Thread through demo flags if you want CLI overrides (see `scripts/learner.py`).

### Trust-region and scale preservation
- Trust-region `tau` uses the same units as `h` and is applied per-row.
- Optional feature-wise scale preservation is available for 2D tensors (config: `scale_preservation`).

### Script usage (developer)
- Learner wiring + demo:
  - Smoothing demo: `py -3 scripts\learner.py --config configs\default.yaml --steps 2 --demo-smoothing`
  - Smoothing overrides: `--k`, `--tau`, `--dim-exponent`, `--td-var-min`, `--disagreement-min`, `--bins`
- Early stopping controls (runtime overrides):
  - `--patience` → overrides `early_stopping.patience`
  - `--eval-interval` → overrides `early_stopping.eval_interval`
  - `--honor-eval-interval` → only update early-stop decision on steps divisible by `eval_interval`
  - Default `--steps` is 12 to allow the minimal controller to trigger with default patience (10)
- Outputs:
  - Metrics → `logs/metrics.jsonl`
  - Events → `logs/events.jsonl` (includes `EarlyStopTriggeredEvent` when `should_stop` becomes true)

### Windows PowerShell tips
- Do not pipe process output to `cat` (alias for Get-Content). Use direct run or `Tee-Object`:
  - `... | Tee-Object -FilePath logs\run.log`
  - Tail logs: `Get-Content logs\metrics.jsonl -Tail 20 -Wait`

### Contribution checklist (Datamutant rules)
- Types on all public functions/classes
- Validate inputs/outputs with assertions (fail fast)
- Entities immutable by default with versions
- Emit events for coordination/observability
- No magic constants: add to config with validation
- Small, single-responsibility functions; compose via registries/APIs


