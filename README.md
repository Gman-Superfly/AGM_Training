## AGM Training Scaffold

Typed, entity-first scaffold for adaptive off-policy RL training with kernel smoothing and early stopping. Uses `uv` for fast, reliable Python envs.

... entity flow inspired by Furlat Abstractions ...

### Prerequisites
- Python >3.10 +
- `uv` package manager (installs virtualenvs, resolves dependencies)

### Quickstart (Windows PowerShell)
```powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
uv venv --python 3.10
uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv run python scripts\learner.py --config configs\default.yaml
uv run python scripts\learner.py --config configs\default.yaml --steps 2 --demo-smoothing
uv run python scripts\learner.py --config configs\default.yaml --steps 2 --demo-smoothing --k 8 --tau 0.2 --dim-exponent 0.5 --td-var-min 0.0 --disagreement-min 0.0
```

### Quickstart (macOS/Linux)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
uv pip install -r requirements.txt
# CPU/MPS or CUDA per platform
uv pip install torch torchvision torchaudio
uv run python scripts/learner.py --config configs/default.yaml
uv run python scripts/learner.py --config configs/default.yaml --steps 2 --demo-smoothing
uv run python scripts/actor.py --config configs/default.yaml --shard 0
```

### Run tests
```bash
pytest
# Optional smoke (Windows):
setx RUN_SMOKE 1
pytest tests/smoke/test_smoke.py
```

### Test suite overview
- **Unit tests**
  - `tests/unit/test_config.py`: Validates config defaults and core constraints (e.g., `k >= 1`, `sigma_min ≤ sigma_max`, `k < batch_size`).
    - Why: Fail fast on misconfiguration; enforce “no magic constants” and cross-field validation.
  - `tests/unit/test_smoothing.py`:
    - `test_kernel_weights_sum_to_one`: Ensures kernel weights are normalized per row.
      - Why: Consensus correctness and numerical stability.
    - `test_trust_region_clip`: Verifies per-row trust-region clipping `||Δh|| ≤ τ`.
      - Why: Safety bounds on updates.
    - `test_alpha_schedule_and_gating_and_sigma_bounds_and_scale_preservation`: Checks cosine schedule monotonicity, sigma clamping, config-driven gating, and featurewise scale preservation.
      - Why: Prevents over-smoothing/drift and guarantees config-governed behavior.
    - Covers `apply_batch_smoothing` diagnostics (gate flag, disagreement, sigma/Δh stats and hist inputs).
      - Why: Observability invariants for telemetry.
  - `tests/unit/test_adaptation.py`:
    - `test_adaptive_smoothing_increase_and_clamps`: Adaptive controller increases `alpha_step`/`k`, decreases `tau`, and clamps within bounds when signals are high.
      - Why: Safe online adaptation within configured limits.
    - `test_adaptive_smoothing_disabled_returns_noop`: No changes when `adapt_enabled=False`.
      - Why: Determinism when adaptation is disabled.
    - `test_adapt_learning_rate_bounds`: LR adaptation respects `[lr_min, lr_max]`.
      - Why: Guard learning rate updates.

- **Lifecycle tests**
  - `tests/life/test_entities_events.py`: Emits `AdaptationAppliedEvent` and verifies JSONL sink write.
    - Why: Event contract and observability pipeline.

- **Smoke tests (opt-in)**
  - `tests/smoke/test_smoke.py`: Starts `scripts/learner.py` and ensures it exits cleanly.
    - Why: Basic wiring/integration check. Enable with `setx RUN_SMOKE 1` then run `pytest tests/smoke/test_smoke.py`.

### Layout
- `configs/default.yaml`: Config entity defaults
- `src/agmlib/`: Config, entities, events, smoothing, early_stop, telemetry
- `scripts/`: learner/actor stubs
- `tests/`: unit, lifecycle, optional smoke

See also: `TESTING_MANUAL.md` for a detailed rationale of modules and tests.

### Smoothing demo & CLI overrides
- Run a tiny smoothing demo (requires torch):
  - Windows: `py -3 scripts\learner.py --config configs\default.yaml --steps 2 --demo-smoothing`
  - macOS/Linux: `python scripts/learner.py --config configs/default.yaml --steps 2 --demo-smoothing`
- Optional overrides (demo only): `--k`, `--tau`, `--dim-exponent`, `--td-var-min`, `--disagreement-min`, `--bins`
- Metrics are written to `logs/metrics.jsonl`; events to `logs/events.jsonl`.

### Early stopping usage & CLI overrides
- Default behavior:
  - `scripts/learner.py` now defaults to `--steps 12` so the minimal early stop can trigger with the default patience (10).
- New flags:
  - `--patience`: override `early_stopping.patience` at runtime
  - `--eval-interval`: override `early_stopping.eval_interval` at runtime
  - `--honor-eval-interval`: only run the early-stop update every `eval_interval` steps (otherwise it runs every step)
- Windows examples:
  - Trigger early stop with defaults: `uv run python scripts\learner.py --config configs\default.yaml`
  - Trigger earlier by lowering patience: `uv run python scripts\learner.py --config configs\default.yaml --steps 5 --patience 3`
  - Honor evaluation cadence: `uv run python scripts\learner.py --config configs\default.yaml --steps 120 --honor-eval-interval --eval-interval 10`

### HMPO ingestion runner (Windows PowerShell)
- Consume HMPO logs and drive AGM controllers, writing controller decisions to JSONL:
  - `uv run python scripts\hmpo_ingest_runner.py --logs .\hmpo\metrics.jsonl --format jsonl --out .\logs\controllers.jsonl`
  - Tail decisions: `Get-Content .\logs\controllers.jsonl -Tail 50 -Wait`

### Further documentation
- HMPO integration guide: `docs/HMPO_INTEGRATION.md`
- CI usage (local and optional GitHub Actions examples): `docs/CI.md`
- Changes by phase/version: `CHANGELOG.md`
- Note: GitHub Actions workflows were removed to keep the repo CI‑agnostic; see `docs/CI.md` to enable CI when needed.

### On‑policy vs Off‑policy RL (some random notes for When to choose)
- Neither is universally better; pick based on constraints.
- Use on‑policy (e.g., PPO/TRPO/A2C) when:
  - Simulator samples are cheap and you want simple, stable updates
  - Rapidly changing policies/data; avoiding replay bias matters
  - Lower sample efficiency is acceptable
- Use off‑policy (e.g., DQN/TD3/SAC) when:
  - Data is expensive; you need high sample efficiency via replay/offline logs
  - Multi‑actor or offline scenarios fit your workflow
  - You can manage stability (target nets, clipping, importance weighting)
- This repo targets off‑policy for sample efficiency; AGM kernel smoothing and early stopping provide stability guardrails.

We know there are new techniques and we will get there slowly as we build with what we know first,  then move to newer stuff when we are sure we fully understand them deeply, this is function first eng repo for us.

### Project Phases & Sources
- Phase 1 — Scaffold, smoothing, early-stop (Complete; v0.1.0)
- Phase 2 — HMPO integration & ingestion (Complete)
- Phase 3 — Advanced controllers & tooling (Complete; v0.2.0)
- Phase 4 — Live rollout, benchmarks, CI & release (Complete)

See:
- Implementation roadmap with phases in `AGM_TRAINING.md` (Implementation Roadmap)
- TODO roadmap with phases in `AGM_ADAPTIVE_PROPERTIES.md` (Implementation TODO Roadmap)
- Execution order in `ACTION_PLAN.md` (Timeline and Order of Execution)
- Detailed changes per phase in `CHANGELOG.md`


