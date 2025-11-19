## AGM Training – Verification & Testing Manual

This manual explains why each major component exists, the invariants it enforces, and how the test suite validates those invariants. It is intended for developers extending or reviewing the system.

### Principles (why we test this way)
- Assert-early, fail-fast: invariants are enforced close to their source (config validation, shapes/ranges, numerical safety).
- Entity-first, event-driven: entities/events are validated for lifecycle and observability.
- No magic constants: thresholds/schedules must come from typed config; tests ensure this contract is respected.
- Numerical guarantees: smoothing weights normalize, trust-region clipping holds, sigma bounds apply, scale preservation works.

---

### Module guide: responsibilities, invariants, and test coverage

#### 1) `src/agmlib/config.py`
- Why it exists: Central, immutable configuration entity (`AGMConfigEntity`) with typed sub-configs. Prevents magic constants and ensures consistent defaults.
- Key invariants:
  - `k` is at least 1 and strictly less than `replay.batch_size`.
  - `sigma_min ≤ sigma_max`.
  - All numeric fields are within documented bounds (via Pydantic constraints).
- How tests validate:
  - `tests/unit/test_config.py`: Loads empty config (defaults) and asserts core ranges. Ensures `load_config` returns an immutable, valid entity and that cross-field checks run.

#### 2) `src/agmlib/entities.py`
- Why it exists: Entity-first modeling for decisions and metrics with IDs, versions, timestamps for traceability.
- Key invariants:
  - Entities carry `ecs_id`, `version`, `created_at` automatically.
  - `SmoothingDecisionEntity` fields constrained (non-negative `alpha_step_used`, `tau_clip`).
  - `EarlyStopDecisionEntity` tracks `should_stop`, `reason`, `patience_left`, and metrics payload.
- How tests validate:
  - `tests/life/test_entities_events.py` uses entities when emitting events; JSONL payloads must serialize and be discoverable.

#### 3) `src/agmlib/events.py`
- Why it exists: Event model for adaptation, early stop, parameter sync, replay updates; plus a simple JSONL emitter for observability.
- Key invariants:
  - Events have unique IDs and timestamps.
  - `emit_event` appends one JSON object per line to sink and prints to stdout.
- How tests validate:
  - `tests/life/test_entities_events.py` writes an `AdaptationAppliedEvent` to a temp sink and verifies presence.

#### 4) `src/agmlib/smoothing.py`
- Why it exists: Core adaptive smoothing mechanics with strict numerical checks.
- Key invariants and their tests:
  - kNN (`compute_knn`):
    - Requires `batch ≥ k+1` and 2D inputs.
    - Distances are finite; diagonal masked; neighbor indices returned.
    - Covered indirectly by downstream tests which rely on valid kNN output.
  - Adaptive sigmas (`adaptive_sigmas`):
    - Sigma derived from kth neighbor distance and clamped to `[sigma_min, sigma_max]`.
    - Validated in `test_alpha_schedule_and_gating_and_sigma_bounds_and_scale_preservation`.
  - Kernel weights (`compute_kernel_weights`):
    - Gaussian weights are finite and row-normalized (sum to ~1).
    - Validated in `test_kernel_weights_sum_to_one`.
  - Consensus (`kernel_consensus`):
    - Applies weights to neighbors of `h` for both 1D and 2D `h`; output is finite.
    - Exercised in end-to-end smoothing test.
  - Trust region update (`trust_region_update`):
    - Computes Δh = α · d^(−dim_exp) · (G − h); clips per-row L2 norm ≤ τ.
    - Validated in `test_trust_region_clip` (per-row norms ≤ τ).
  - Scale preservation (`featurewise_scale_preservation`):
    - Per-dimension std of `after` is adjusted to match `before` (for 2D `h`).
    - Validated in end-to-end smoothing test (orig vs new std close).
  - Gating and schedule:
    - `schedule_alpha_step` follows config (cosine/constant), monotone non-increasing across decay.
    - `should_apply_smoothing_decision` uses ONLY config thresholds (`td_var_min`, `disagreement_min`).
    - Validated in `test_alpha_schedule_and_gating_and_sigma_bounds_and_scale_preservation`.
  - End-to-end flow (`apply_batch_smoothing`):
    - Returns updated `h` and diagnostics: gate flag, disagreement, sigma stats, Δh stats, and raw tensors for histograms.
    - Validated in the same unit test.

#### 5) `src/agmlib/adaptation.py`
- Why it exists: Online adaptation controller to gently tune smoothing parameters (and a minimal LR adapter) based on stability signals.
- Key invariants:
  - `AdaptiveSmoothingController.adapt` mutates only a runtime copy (`cfg_run`) and respects all configured min/max clamps for `alpha_step`, `k`, `tau`.
  - No-ops if `adapt_enabled=False`.
  - `adapt_learning_rate` keeps LR in `[lr_min, lr_max]`.
- How tests validate:
  - `tests/unit/test_adaptation.py` includes:
    - Increase/clamping behavior under high variance/disagreement.
    - No-op when disabled.
    - LR clamped to bounds.

#### 6) `src/agmlib/early_stop.py`
- Why it exists: Minimal early stopping controller based on TD stats, Q stability, and eval reward trend proxy.
- Key invariants:
  - Patience countdown respects hysteresis; optional weighted cost mode governed by config.
  - Decision returned as `EarlyStopDecisionEntity` with metrics payload.
- How it is validated:
  - Smoke & runtime: exercised in `scripts/learner.py` loop; lifecycle covered by event emission test.
  - Extend with targeted unit tests if logic grows (trend tests, etc.).

#### 7) `src/agmlib/replay.py`
- Why it exists: Simple PER implementations (array-based and segment-tree) to support training and priority updates without external deps.
- Key invariants:
  - `alpha`, `beta` ∈ [0,1]; capacities positive; sampling probabilities normalized.
  - IS weights computed and normalized to ≤1.
- How it is validated:
  - Exercised in `scripts/learner.py` (training path). Add unit tests if replay becomes a focus (e.g., tree correctness) beyond the demo scope.

#### 8) `src/agmlib/dqn.py`
- Why it exists: Minimal DQN network and double-DQN target utility for the learning demo.
- Key invariants:
  - Torch optionality: fail fast if used without torch.
  - Target computation uses online argmax and target network gather.
- How it is validated:
  - Indirectly via `scripts/learner.py` demo training loop and smoke test.

#### 9) `src/agmlib/telemetry.py`
- Why it exists: Append-only metrics logging and simple histogram utility that accepts tensors/arrays.
- Key invariants:
  - JSONL write per call; histogram returns lists and min/max.
- How it is validated:
  - Used in demo; inputs exercised in unit tests that prepare histogram-ready outputs.

#### 10) `scripts/learner.py`
- Why it exists: Integration harness that wires config, smoothing, adaptation, early stop, telemetry, and events into a minimal train/eval loop.
- Key invariants:
  - Uses runtime config copy (`cfg_run`) for safe adaptation.
  - Emits `AdaptationAppliedEvent`, `EarlyStopTriggeredEvent`, `ReplayUpdateEvent`, `ParameterSyncEvent`.
  - Logs sigma/Δh histograms and smoothing application rate periodically.
- How it is validated:
  - Smoke test `tests/smoke/test_smoke.py` confirms the script runs with defaults and exits success.
  - Unit and lifecycle tests exercise the building blocks it orchestrates.

---

### Tests overview (what & why)

1) `tests/unit/test_config.py`
   - Validates default config loading and key constraints.
   - Why: Catch misconfigurations early; ensure defaults are safe and consistent with action plan.

2) `tests/unit/test_smoothing.py`
   - `test_kernel_weights_sum_to_one`: Weight normalization.
   - `test_trust_region_clip`: Δh clipping bound.
   - `test_alpha_schedule_and_gating_and_sigma_bounds_and_scale_preservation`: Schedule monotonicity, sigma bounds, config-driven gating, scale preservation, diagnostics presence.
   - Why: Guarantees numerically safe smoothing behavior with config-driven gates.

3) `tests/unit/test_adaptation.py`
   - Verifies adaptive smoothing increases/decreases and clamps; no-op when disabled; LR adaptation respects bounds.
   - Why: Safety and determinism of online adaptation.

4) `tests/life/test_entities_events.py`
   - Emits an adaptation event and validates JSONL sink.
   - Why: Ensures observability pipeline works end-to-end and entities serialize.

5) `tests/smoke/test_smoke.py` (opt-in)
   - Starts `scripts/learner.py` with default config and asserts clean exit.
   - Why: Quick integration health check without heavy dependencies.

---

### Extending tests (how to add and what to assert)
- Add invariants first: state in code via assertions and model constraints.
- For new modules or features:
  - Unit-test pure functions (shapes, ranges, monotonicity, numerical guards).
  - Lifecycle-test entities/events when crossing boundaries or producing telemetry.
  - Update this manual with “why” for each new test.

---

### Running tests
- Full suite: `py -3 -m pytest -v`
- Smoke only (Windows PowerShell):
  - `setx RUN_SMOKE 1`
  - `py -3 -m pytest tests\smoke\test_smoke.py -v`


