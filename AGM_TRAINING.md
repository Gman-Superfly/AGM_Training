## AGM Training: Adaptive AGM for Off-Policy RL with Distributed Multi‑GPU and Early Stopping

### Executive Summary

This document defines the architecture and methodology for an adaptive, distributed reinforcement learning system that combines:
- Off‑policy RL (Double DQN + Prioritized Replay) with decentralized multi‑GPU execution
- Multi‑criteria early stopping grounded in Q‑learning signals
- Kernel‑weighted drift smoothing with adaptive bandwidths and a Free Energy Principle (FEP) objective to reduce off‑policy noise and oscillations
- An entity‑first, event‑driven design aligned with Datamutant development rules (assertions, types, immutability, composability)

The system targets faster and more stable convergence, robust performance under non‑stationarity, and scalable throughput across multiple GPUs.

---

## System Goals

- Improve training stability and sample efficiency for off‑policy RL via adaptive smoothing and principled early stopping
- Scale data collection and learning across multiple GPUs with low synchronization overhead
- Maintain observability and auditability via events and immutable entities
- Keep numerical behavior stable (noisy targets, high‑dim data, distributed staleness)

---

## Core Components

### Off‑Policy RL Backbone

- **Algorithm**: Double DQN with target networks and Prioritized Experience Replay (PER)
- **Why**: Off‑policy sample reuse, stable targets (Double DQN), and variance reduction (PER)
- **Key signals**:
  - TD error and variance
  - Q‑value convergence metrics
  - Validation policy reward curve

### Distributed Multi‑GPU Architecture (Ape‑X style)

- **Actors (per GPU or CPU)**: Interact with environments, push transitions to a shared or sharded replay buffer
- **Learners (GPU)**: Sample batches, compute updates, periodically broadcast parameters to actors
- **Synchronization**:
  - Parameter push: learner → actors at interval K
  - Replay buffer: centralized or sharded; prioritization maintained per shard; periodic merges optional
- **Frameworks**: PyTorch (DistributedDataParallel optional), Ray/RLlib for orchestration (optional but recommended)

### Kernel‑Weighted Drift Smoother with FEP

We introduce a batch‑level smoothing step to dampen off‑policy noise in TD targets or latent embeddings via a consensus‑style update with adaptive kernels and an FEP objective.

- Iterative update:
  - \( h^{(t+1)} = h^{(t)} + \alpha_{\text{step}} \cdot \Delta t \cdot \big(G[h^{(t)}] - h^{(t)}\big) + \eta \)
  - \( G[h] \): kernel‑weighted neighborhood consensus
  - \( \alpha_{\text{step}} \): adaptive smoothing step size (not PER alpha)
  - \( \Delta t = d^{-\text{dim\_exponent}} \): dimension‑aware step
  - \( \eta \): optional exploration noise

- Adaptive kernel bandwidths:
  - \( \sigma_i = \| h_i - h_i^{(k)} \|_2 \) using k‑th nearest neighbor
  - Gaussian kernel: \( K(h_i, h_j) = \exp\big(-\|h_i-h_j\|^2 / (2\sigma_i^2)\big) \)

- Free Energy Principle (FEP):
  - \( F[H] = U[H] - T \cdot S[H] \)
  - Energy: \( U[H] = \frac{1}{2}\sum_i \| h_i - \delta_i \|^2 \) (prediction error to targets)
  - Entropy: \( S[H] = -\sum_{i,j} p(j|i)\log p(j|i) \), with \( p(j|i) \propto K(h_i, h_j) \)
  - Effective temperature arises from \( \alpha_{\text{step}} \) and \( \sigma_i \) interplay

- Naming disambiguation (to avoid conflicts):
  - `alpha_step`: smoother step size used in the consensus update
  - `dim_exponent`: exponent used in \( \Delta t = d^{-\text{dim\_exponent}} \)
  - `replay.alpha`: PER priority exponent; unrelated to smoothing `alpha_step`
  - Code–equation mapping:
    - Equation symbol \(\alpha\) in \(\Delta t = d^{-\alpha}\) is implemented as `dim_exponent`
    - Equation symbol \(\alpha\) in the consensus update step size is implemented as `alpha_step`
    - PER priority exponent \(\alpha_{PER}\) is `replay.alpha`

- Scale preservation:
  - Normalize feature‑wise scale to prevent variance collapse:
    - \( h \leftarrow h \times (\sigma_{\text{original}} / \sigma_{\text{current}}) \)
  - Optional stochastic noise η with std `kernel_smoothing.noise_std` added pre‑clipping to encourage exploration

- Application points:
  - Smoothing TD targets in latent space (low risk, high reward)
  - Smoothing encoder latents for Q‑head stability
  - Small regularizer on Q‑head toward \( G[h] \) with trust‑region bound

### Early Stopping (Q‑Grounded, Multi‑Criteria)

- **Signals**:
  - EMA of TD error mean/variance
  - Q‑value convergence stability (e.g., relative change, Lipschitz proxy over neighbors)
  - Validation policy reward trend and plateau detection
- **Decision**:
  - Periodic evaluation (every M updates)
  - Multi‑criteria cost with patience and hysteresis:
    - Example: cost = \( 0.3 \times \text{TD distortion} + 0.5 \times (1 - \text{local structure}) + 0.2 \times \text{plateau risk} \)
  - Stop when:
    - Convergence is high and stable, and validation reward plateaus by small threshold over N evals
    - Or TD variance remains very low with no improvement

---

## Detailed Design

### Data Flow

1. Actors generate transitions `(s, a, r, s', done, info)` and push to replay (PER).
2. Learner samples mini‑batches; encodes states to latent \( z \in \mathbb{R}^d \).
3. Compute TD targets (Double DQN) and TD errors.
4. Apply kernel‑weighted drift smoother to either:
   - TD targets in latent neighborhoods, or
   - Latent embeddings \( z \) (then compute TD on smoothed latents).
5. Optimize Q‑network with standard loss plus optional small consensus regularizer.
6. Update priorities; periodically push weights to actors.
7. Periodically evaluate policy → early stopping controller decides continue/stop.

### Kernel Smoothing (Batch‑Level)

- Neighborhoods:
  - Use kNN within the current mini‑batch (k ≈ 8–32) for O(batch log batch)
  - Compute distances in latent space (d ≈ 32–128), L2‑normalized
  - Pure‑PyTorch path: `torch.cdist` + `topk` (optionally computed in blocks for large batches)

- Consensus:
  - For each \( i \): compute \( \sigma_i \) from kth neighbor
  - Weights \( w_{ij} \propto \exp\big(-\|z_i - z_j\|^2 / (2\sigma_i^2)\big) \), normalized over j
  - \( G[h]_i = \sum_j w_{ij} h_j \)

- Update:
  - \( \Delta h_i = \alpha_{\text{step}} \cdot \Delta t \cdot (G[h]_i - h_i) \)
  - Trust region: clip \( \|\Delta h_i\| \le \tau \) (\(\tau\) is in the same units as \(h\))
  - Gate by uncertainty:
    - Apply only when local TD variance high or neighbor disagreement high (thresholds are configurable; no hard‑coded constants)
  - Anneal \( \alpha_{\text{step}} \) over training (e.g., cosine decay)
  - Scheduler: cosine decay controlled by `kernel_smoothing.alpha_step_decay_steps`

- Numerical safeguards:
  - Clamp \( \sigma_i \in [\sigma_{\min}, \sigma_{\max}] \)
  - Add small eps to denominators
  - Verify non‑empty neighborhoods via asserts
  - Optional feature‑wise scale preservation (config: `kernel_smoothing.scale_preservation`)

### Free Energy Objective (Optional Augmentation)

- Compute:
  - \( U = \frac{1}{2}\|h - \delta\|^2 \) (per batch)
  - \( S = -\sum_{i} \sum_{j} p(j|i) \log p(j|i) \)
  - \( F = U - T \cdot S \)
- Use as a monitoring metric and/or add a small term \( \lambda_F \cdot F \) to the loss
- Implemented as `free_energy(h, delta, z, knn, sigmas, T)` for diagnostics; wire into loss optionally
- Tune \( T \) implicitly via \( \alpha_{\text{step}} \), \( \sigma_i \), or expose as a parameter

### Early Stopping Controller

- Metrics (per eval window):
  - EMA TD error mean/variance (window W1)
  - Q‑convergence stability (relative change over W2)
  - Validation reward moving average and slope (W3)
  - Optional: Mann–Kendall trend on reward to guard against noise
- Logic:
  - Patience N evals; hysteresis bands (e.g., 2–5% tolerance)
  - Combine into a decision entity:
    - `EarlyStopDecision(status: Continue|Stop, reason: str, metrics: EarlyStopMetrics)`
- Distributed aggregation:
  - Collect metrics from learners/actors; coordinator averages with robust statistics (median/trimmed mean)

---

## Entities, Events, and Registries (Datamutant‑Aligned)

### Typed Entities (Pydantic or dataclasses)

- `AGMConfigEntity`
  - Replay, batch, kNN: `k`, `sigma_min`, `sigma_max`, `alpha_step_init`, `alpha_step_schedule`, `dim_exponent`, `trust_region_tau`, `latent_dim`
  - Early stop: `eval_interval`, `patience`, `hysteresis_pct`, `td_ema_beta`
  - Distributed: `param_sync_interval`, `replay_shards`, `actors_per_gpu`
  - Validate ranges on init

- `AGMMetricsEntity`
  - `convergence_rate`, `mean_spread`, `oscillation_strength`, `iterations_used`, `timestamp`
  - For RL: `td_mean`, `td_var`, `q_rel_change`, `val_reward`

- `SmoothingDecisionEntity`
  - `applied: bool`, `alpha_step_used`, `tau_clip`, `gated_by_uncertainty: bool`, `fep_value`, `sigma_stats`

- `EarlyStopDecisionEntity`
  - `should_stop: bool`, `reason: str`, `patience_left: int`, `metrics: EarlyStopMetrics`

- `PopulationMetricsEntity` (if using population training)
  - Diversity stats, strategy entropy, convergence histograms

All entities must assert required invariants and version numbers; emit events on creation/application.

### Events

- `AdaptationAppliedEvent(subject_id, step, decision: SmoothingDecisionEntity, pre_metrics, post_metrics)`
- `EarlyStopTriggeredEvent(subject_id, step, decision: EarlyStopDecisionEntity)`
- `ParameterSyncEvent(learner_id, step, checksum, model_version)`
- `ReplayUpdateEvent(shard_id, count, priority_stats)`

### Registries

- Entity registry for versioned records; callable registry for operations (e.g., `"compute_knn_kernel"`, `"apply_consensus_update"`, `"early_stop_check"`)

---

## Training Loop (High‑Level Pseudocode)

```python
# Pseudocode (PyTorch-first, typed, asserts implied)
for step in range(max_steps):
    batch = replay.sample(B)  # PER
    z = encoder(s)            # latent features, L2-normalized
    
    with torch.no_grad():
        q_next = target_q(next_s).max(1).values
        td_target = r + gamma * (1 - done) * q_next

    q_pred = q(s).gather(1, a)
    td_error = td_target - q_pred

    # Optional: smooth TD targets or z via kernel consensus
    if should_apply_smoothing(td_error, config):
        knn = compute_knn(z, k=config.k)  # torch.cdist + topk
        G = kernel_consensus(h=td_target, z=z, knn=knn, sigmas=adaptive_sigmas(z, knn, config))
        td_target = trust_region_update(h=td_target, G=G, alpha_step=schedule_alpha_step(step), tau=config.tau)

    loss = mse(q_pred, td_target.detach()) + lambda_reg * consensus_reg(q_pred, G)  # small reg optional
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(q.parameters(), max_norm)
    optimizer.step()

    update_priorities(td_error)
    if step % param_sync_interval == 0:
        broadcast_params_to_actors(q.state_dict())

    if step % eval_interval == 0:
        eval_metrics = evaluate_policy(policy, N_episodes)
        decision = early_stop_controller.update(td_error_stats, q_stability, eval_metrics)
        emit(EarlyStopDecisionEntity(...))
        if decision.should_stop:
            break
```

---

## Numerical Stability and Performance

- **Compute bounds**:
  - Batch‑local kNN only; k≈8–32; latent dim 32–128
  - Pure‑PyTorch kNN: `torch.cdist` + `topk`; for large batches, compute pairwise distances in blocks to control memory
- **Gating**:
  - Apply smoothing only under high local uncertainty/variance (configurable; no hard‑coded thresholds)
  - Anneal \( \alpha_{\text{step}} \); trust‑region clipping on \( \Delta h \)
- **Avoid over‑smoothing**:
  - Keep Double DQN/target networks
  - Consider importance‑weighted kernels (IS ratios) if behavior policy drift is large
- **Noisy signals**:
  - EMA smoothing for TD/error metrics; hysteresis for decisions
  - Minimum dwell time between adaptations to avoid oscillations

---

## Risks and Mitigations

- **Kernel cost**: Use batch‑local kNN, approximate search, and sub‑batch blocks
- **Value function bias**: Gate smoothing by uncertainty; trust region; keep Double DQN
- **Distributed staleness**: Local kernels; coarse parameter sync; avoid replay‑wide global kernels
- **Early stop false positives**: Combine multiple signals; apply patience and statistical trend checks
- **High‑dim curse**: Operate in compact latent; \( \Delta t = d^{-\text{dim\_exponent}} \)

---

## Implementation Roadmap

- **Phase 1: Backbone + Metrics**
  - Implement Double DQN + PER, distributed actors/learner
  - Instrument TD error, Q stability, validation evals; add entities and events

- **Phase 2: Kernel Smoothing MVP**
  - Batch‑local latent smoothing on TD targets
  - kNN with adaptive \( \sigma_i \); trust region; \( \alpha_{\text{step}} \) annealing
  - Pure‑PyTorch implementation (no FAISS), guarded by uncertainty; logging + events

- **Phase 3: Early Stopping Controller**
  - Multi‑criteria with EMA, hysteresis, patience; distributed aggregation
  - Wire `EarlyStopDecisionEntity` and handlers

- **Phase 4: FEP Monitoring/Regularization**
  - Compute F[H] metrics; optional small regularization term
  - Sensitivity analysis

- **Phase 5: Scale Out**
  - Multi‑GPU learners; parameter sync
  - Sharded replay; throughput profiling and optimization

- **Phase 6: Population Extensions (Optional)**
  - Multiple agents with diverse AGM biases; coordination and evolution

---

## Testing Strategy

- **Unit tests**:
  - kNN, kernel weights, sigma bounds, consensus step invariants
  - Trust‑region clipping, alpha_step schedule, scale preservation
  - Early stopping decision logic (patience/hysteresis)

- **Lifecycle tests**:
  - Entity creation/versioning; event emission/handling
  - Replay priority updates; parameter sync events

- **Integration tests**:
  - CartPole/Atari small‑scale runs: baseline vs smoothing+early stop
  - Metrics: time‑to‑target, area‑under‑curve, variance of TD/error, oscillation rates

- **Regression guards**:
  - Seeded deterministic runs; tolerance bands for key metrics

---

## Configuration (Example)

```yaml
# AGMConfigEntity (example)
replay:
  batch_size: 256
  prioritized_replay: true
  alpha: 0.6
  beta0: 0.4

kernel_smoothing:
  enabled: true
  k: 16
  sigma_min: 1e-3
  sigma_max: 1.0
  alpha_step_init: 0.15
  alpha_step_schedule: cosine
  alpha_step_decay_steps: 100000
  dim_exponent: 0.5
  trust_region_tau: 0.1
  latent_dim: 64
  gate_by_td_variance: true
  td_var_min: 0.0
  disagreement_min: 0.0
  scale_preservation: false

early_stopping:
  eval_interval: 5000
  patience: 10
  hysteresis_pct: 0.03
  td_ema_beta: 0.9
  plateau_threshold: 0.5

distributed:
  actors_per_gpu: 4
  learners: 1
  param_sync_interval: 1000
  replay_shards: 4
```

---

## Windows (PowerShell) Quickstart — using uv

```powershell
# Install uv (if not installed)
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

# Create virtual env with Python 3.10 (creates .venv)
uv venv --python 3.10

# Option A: activate
. .\.venv\Scripts\Activate.ps1

# Option B: use `uv run` without activating (shown below)

# Install base requirements
uv pip install -r requirements.txt

# Optional: install PyTorch CUDA 12.1 wheels
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: RL stack
uv pip install "ray[rllib]"

# Run learner and actors (example layout)
$env:CUDA_VISIBLE_DEVICES="0" ; uv run python scripts\learner.py --config configs\default.yaml
$env:CUDA_VISIBLE_DEVICES="1" ; uv run python scripts\actor.py --config configs\default.yaml --shard 0
$env:CUDA_VISIBLE_DEVICES="2" ; uv run python scripts\actor.py --config configs\default.yaml --shard 1
```

---

## macOS (zsh) Setup — using uv

Note: These are possible setups only; project scaffolding and commands may change as we finish implementation.

```zsh
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual env
uv venv --python 3.10

# Option A: activate
source .venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Optional: torch CPU/MPS
uv pip install torch torchvision torchaudio

# Optional: RL stack
uv pip install "ray[rllib]"

# Run
uv run python scripts/learner.py --config configs/default.yaml
uv run python scripts/actor.py --config configs/default.yaml --shard 0
uv run python scripts/actor.py --config configs/default.yaml --shard 1
```

---

## Linux (bash) Setup — using uv

Note: These are possible setups only; project scaffolding and commands may change as we finish implementation.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
uv venv --python 3.10

# Option A: activate
source .venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Optional: CUDA 12.1 wheels shown; adjust index-url per your CUDA toolkit, or omit for CPU-only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: RL stack
uv pip install "ray[rllib]"

# Run
CUDA_VISIBLE_DEVICES=0 uv run python scripts/learner.py --config configs/default.yaml
CUDA_VISIBLE_DEVICES=1 uv run python scripts/actor.py --config configs/default.yaml --shard 0
CUDA_VISIBLE_DEVICES=2 uv run python scripts/actor.py --config configs/default.yaml --shard 1
```

---

## Licensing

- Project licensed under MIT (see `LICENSE`).

---

## Appendix: Notation and Key Equations

- Consensus update:
  \[
  h^{(t+1)} = h^{(t)} + \alpha_{\text{step}} \cdot \Delta t \cdot \big(G[h^{(t)}] - h^{(t)}\big) + \eta,\quad
  G[h]_i = \sum_{j} \frac{\exp\left(-\frac{\|z_i-z_j\|^2}{2\sigma_i^2}\right)}{\sum_{k} \exp\left(-\frac{\|z_i-z_k\|^2}{2\sigma_i^2}\right)} h_j
  \]
- Free Energy:
  \[
  F[H] = \frac{1}{2}\sum_i \|h_i - \delta_i\|^2 - T \cdot \sum_{i,j} p(j|i)\log p(j|i)
  \]
- Dimension‑aware step: \( \Delta t = d^{-\text{dim\_exponent}} \)

---

## Phase Status Snapshot
- Phase 1 — Scaffold, smoothing, early-stop: Complete (see v0.1.0 in `CHANGELOG.md`)
- Phase 2 — HMPO integration & ingestion: Complete
- Phase 3 — Advanced controllers & tooling: Complete (see v0.2.0 in `CHANGELOG.md`)
- Phase 4 — Live rollout, benchmarks, CI & release: Complete

Reference phase planning:
- Implementation Roadmap (this document)
- `AGM_ADAPTIVE_PROPERTIES.md` — Implementation TODO Roadmap (Phases 1–4)
- `ACTION_PLAN.md` — Timeline and Order of Execution