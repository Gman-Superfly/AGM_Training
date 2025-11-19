import argparse
from pathlib import Path
from typing import Dict

from agmlib.config import load_config
from agmlib.early_stop import EarlyStopController
from agmlib.entities import SmoothingDecisionEntity
from agmlib.events import AdaptationAppliedEvent, EarlyStopTriggeredEvent, ParameterSyncEvent, ReplayUpdateEvent, emit_event
from agmlib.telemetry import MetricsLogger, histogram
from agmlib.smoothing import (
    schedule_alpha_step,
    compute_knn,
    adaptive_sigmas,
    kernel_consensus,
    trust_region_update,
    featurewise_scale_preservation,
    average_disagreement,
    should_apply_smoothing_decision,
    apply_batch_smoothing,
)
from agmlib.replay import PrioritizedReplayBuffer, SegmentTreePER, Transition
from agmlib.dqn import TinyDQN, DQNHyperParams, double_dqn_targets
from agmlib.adaptation import AdaptiveSmoothingController, MultiScaleEma, adapt_learning_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--demo-smoothing", action="store_true", help="Run a tiny in-process smoothing demo if torch is available")
    # CLI overrides for demo tuning
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--dim-exponent", type=float, default=None)
    parser.add_argument("--td-var-min", type=float, default=None)
    parser.add_argument("--disagreement-min", type=float, default=None)
    # Early stop and evaluation cadence overrides (optional)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--honor-eval-interval", action="store_true", help="Only run early-stop update every eval_interval steps")
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # Apply optional early stop overrides using an immutable copy for runtime
    cfg_run = cfg.model_copy(deep=True)
    if args.patience is not None:
        object.__setattr__(cfg_run.early_stopping, "patience", int(args.patience))
    if args.eval_interval is not None:
        object.__setattr__(cfg_run.early_stopping, "eval_interval", int(args.eval_interval))

    metrics = MetricsLogger()
    early_stop = EarlyStopController(cfg_run)

    # Minimal DQN wiring for demonstration (CPU by default if torch available)
    try:
        import importlib
        torch = importlib.import_module("torch")
        T = torch
        import gymnasium as gym
        obs_dim, num_actions = 8, 4
        # Infer shape from env
        env = gym.make(args.env_id)
        eval_env = gym.make(args.env_id)
        obs_space = env.observation_space
        act_space = env.action_space
        assert hasattr(obs_space, "shape") and len(obs_space.shape) == 1, "Only flat obs supported in demo"
        obs_dim = int(obs_space.shape[0])
        assert hasattr(act_space, "n"), "Only discrete actions supported in demo"
        num_actions = int(act_space.n)
        online = TinyDQN(obs_dim, num_actions)
        target = TinyDQN(obs_dim, num_actions)
        target.load_state_dict(online.state_dict())
        opt = T.optim.Adam(online.parameters(), lr=float(cfg_run.training.learning_rate))
        hp = DQNHyperParams()
        # Tiny in-process synthetic replay
        # Choose segment-tree PER for efficiency
        rb = SegmentTreePER(capacity=10000, alpha=cfg.replay.alpha, beta0=cfg.replay.beta0)
        # Seed buffer with random transitions
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(512):
            s = rng.standard_normal(size=(obs_dim,)).astype("float32")
            a = int(rng.integers(0, num_actions))
            r = float(rng.normal())
            ns = rng.standard_normal(size=(obs_dim,)).astype("float32")
            d = bool(rng.random() < 0.1)
            rb.push(Transition(s, a, r, ns, d))
    except Exception:
        torch = None

    # Collect real experience and train; fall back to synthetic loop if torch missing
    if torch is not None:
        assert env is not None and eval_env is not None
        global_step = 0
        import numpy as np
        ema_td_mean = 0.0
        ema_td_var = 0.0
        ema_beta = float(cfg_run.early_stopping.td_ema_beta)
        prev_q = None
        # Multi-scale metrics and adaptive controller
        td_var_ms = MultiScaleEma(short=0.6, medium=0.85, long=0.97)
        dis_ms = MultiScaleEma(short=0.6, medium=0.85, long=0.97)
        fep_ms = MultiScaleEma(short=0.6, medium=0.85, long=0.97)
        adaptor = AdaptiveSmoothingController(cfg_run)
        # Telemetry: smoothing application rate
        smooth_applied = 0
        smooth_total = 0
        for ep in range(args.train_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                eps = float(args.epsilon)
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    with T.no_grad():
                        q_vals = online(T.tensor(obs, dtype=T.float32).unsqueeze(0))
                        action = int(T.argmax(q_vals, dim=1).item())
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                rb.push(Transition(np.asarray(obs, dtype=np.float32), action, float(reward), np.asarray(next_obs, dtype=np.float32), done))
                obs = next_obs
                global_step += 1

                if len(rb) >= cfg.replay.batch_size:
                    batch, idxs, iw = rb.sample(cfg.replay.batch_size)
                    s = T.tensor(batch["state"], dtype=T.float32)
                    a = T.tensor(batch["action"], dtype=T.int64)
                    r = T.tensor(batch["reward"], dtype=T.float32)
                    ns = T.tensor(batch["next_state"], dtype=T.float32)
                    d = T.tensor(batch["done"], dtype=T.float32)
                    w = T.tensor(iw, dtype=T.float32)

                    # Compute latents and TD targets
                    z = online.encode(s)
                    td_tgt = double_dqn_targets(online, target, ns, r, d, hp.gamma)
                    # Smooth targets using previous EMA TD variance as gating signal
                    td_tgt_sm, diag = apply_batch_smoothing(
                        h=td_tgt,
                        z=z,
                        step=global_step,
                        td_var=float(ema_td_var),
                        config=cfg_run,
                    )
                    # Track smoothing application rate
                    smooth_total += 1
                    if bool(diag.get("gate", False)):
                        smooth_applied += 1
                    q_pred = online(s).gather(1, a.view(-1, 1)).squeeze(1)
                    loss = ((w * (q_pred - td_tgt_sm.detach()) ** 2).mean())
                    opt.zero_grad()
                    loss.backward()
                    T.nn.utils.clip_grad_norm_(online.parameters(), max_norm=hp.max_grad_norm)
                    opt.step()

                    # Update priorities and soft target
                    td = (td_tgt_sm.detach() - q_pred.detach())
                    td_err = td.abs().cpu().numpy()
                    rb.update_priorities(idxs, td_err + 1e-6)
                    # Emit replay update event
                    try:
                        import numpy as _np
                        pr_stats = {
                            "min": float(_np.min(td_err).item()) if td_err.size > 0 else 0.0,
                            "mean": float(_np.mean(td_err).item()) if td_err.size > 0 else 0.0,
                            "max": float(_np.max(td_err).item()) if td_err.size > 0 else 0.0,
                        }
                        emit_event(
                            ReplayUpdateEvent(
                                shard_id="local",
                                count=int(len(idxs)),
                                priority_stats=pr_stats,
                            )
                        )
                    except Exception:
                        pass
                    tau = 0.005
                    with T.no_grad():
                        for p_t, p_o in zip(target.parameters(), online.parameters()):
                            p_t.data.copy_((1 - tau) * p_t.data + tau * p_o.data)
                    # Emit parameter sync event with simple checksum
                    try:
                        with T.no_grad():
                            checksum_val = 0.0
                            for p in target.parameters():
                                checksum_val += float(T.sum(T.abs(p)).item())
                        emit_event(
                            ParameterSyncEvent(
                                learner_id="learner-0",
                                step=global_step,
                                checksum=f"{checksum_val:.6f}",
                                model_version=str(global_step),
                            )
                        )
                    except Exception:
                        pass

                    # Update EMA TD stats and Q stability
                    with T.no_grad():
                        td_mean_now = float(td.mean().item())
                        td_var_now = float(T.var(td, unbiased=False).item())
                        ema_td_mean = ema_beta * ema_td_mean + (1.0 - ema_beta) * td_mean_now
                        ema_td_var = ema_beta * ema_td_var + (1.0 - ema_beta) * td_var_now
                        if prev_q is not None:
                            rel = T.mean(T.abs(q_pred - prev_q) / (T.abs(prev_q) + 1e-6))
                            q_stab_rel = float(rel.item())
                        else:
                            q_stab_rel = 0.0
                        prev_q = q_pred.detach()

                        # Update multi-scale metrics
                        td_var_ms_new = td_var_ms.update(ema_td_var)
                        dis_ms_new = dis_ms.update(float(diag.get("avg_disagreement", 0.0)))
                        fep_val = None
                        try:
                            from agmlib.smoothing import free_energy
                            knn = compute_knn(z, cfg_run.kernel_smoothing.k)
                            sigmas = adaptive_sigmas(z, knn, cfg_run)
                            fep_val = free_energy(h=td_tgt_sm.detach(), delta=td_tgt.detach(), z=z.detach(), knn=knn, sigmas=sigmas, temperature=1.0)
                        except Exception:
                            pass
                        fep_ms_new = fep_ms.update(float(fep_val) if fep_val is not None else 0.0)
                        td_var_ms, dis_ms, fep_ms = td_var_ms_new, dis_ms_new, fep_ms_new

                        # Adaptive controller applies gentle online adjustments
                        decision = adaptor.adapt(
                            cfg_run=cfg_run,
                            step=global_step,
                            ema_td_var=td_var_ms.short_val,
                            avg_disagreement=dis_ms.short_val,
                            fep_value=fep_val,
                        )
                        emit_event(AdaptationAppliedEvent(subject_id=str(decision.ecs_id), step=global_step, decision={"alpha_step": decision.alpha_step_used, "k": cfg_run.kernel_smoothing.k, "tau": cfg_run.kernel_smoothing.trust_region_tau}, pre_metrics={"ema_td_var_s": td_var_ms.short_val, "dis_s": dis_ms.short_val}, post_metrics={}))

                        # Adapt learning rate based on stability (inverse of short-term td_var)
                        try:
                            stability = 1.0 / (1.0 + float(td_var_ms.short_val))
                            new_lr = adapt_learning_rate(cfg_run=cfg_run, stability=stability)
                            for g in opt.param_groups:
                                g["lr"] = new_lr
                        except Exception:
                            pass

                        # Periodic telemetry: histograms and application rate
                        if (global_step % max(1, cfg_run.early_stopping.eval_interval // 10)) == 0:
                            try:
                                metrics.log(
                                    {
                                        "step": global_step,
                                        "sigma_hist": histogram(diag["sigmas"].squeeze(1)),
                                        "delta_h_hist": histogram(diag["delta_h"]),
                                        "avg_disagreement": diag.get("avg_disagreement", 0.0),
                                        "gate": bool(diag.get("gate", False)),
                                        "sigma_stats": diag.get("sigma_stats", {}),
                                        "delta_h_stats": diag.get("delta_h_stats", {}),
                                        "smooth_apply_rate": float(smooth_applied / max(1, smooth_total)),
                                    }
                                )
                            except Exception:
                                pass

            # Periodic evaluation episodes feeding early stop
            eval_rewards = []
            for _ in range(args.eval_episodes):
                o, _ = eval_env.reset()
                done_eval = False
                total_r = 0.0
                while not done_eval:
                    with T.no_grad():
                        q_vals = online(T.tensor(o, dtype=T.float32).unsqueeze(0))
                        act = int(T.argmax(q_vals, dim=1).item())
                    o, r, term, trunc, _ = eval_env.step(act)
                    done_eval = bool(term or trunc)
                    total_r += float(r)
                eval_rewards.append(total_r)
            avg_val_reward = float(np.mean(eval_rewards)) if len(eval_rewards) > 0 else 0.0

            # Update early stop (using placeholders for td stats & stability)
            stop_decision = early_stop.update({"mean": float(ema_td_mean), "var": float(ema_td_var)}, {"rel_change": float(q_stab_rel) if 'q_stab_rel' in locals() else 0.0}, {"val_reward": avg_val_reward, "improvement": 0.0})
            # Save best checkpoint
            if ep == 0 or (len(eval_rewards) > 0 and avg_val_reward >= max(eval_rewards)):
                ckpt = Path("logs") / "checkpoints"
                ckpt.mkdir(parents=True, exist_ok=True)
                T.save(online.state_dict(), ckpt / f"dqn_{args.env_id}_best.pt")
            metrics.log({"episode": ep, "val_reward": avg_val_reward, "early_stop": stop_decision.should_stop, "patience_left": stop_decision.patience_left})
            if stop_decision.should_stop:
                emit_event(EarlyStopTriggeredEvent(subject_id=str(stop_decision.ecs_id), step=global_step, decision={"reason": stop_decision.reason}))
                break

        return

    # Fallback minimal placeholder loop when torch not available
    for step in range(args.steps):
        td_stats: Dict[str, float] = {"mean": 0.1 / (step + 1), "var": 0.01 / (step + 1)}
        q_stab: Dict[str, float] = {"rel_change": 0.05 / (step + 1)}
        eval_m: Dict[str, float] = {"val_reward": float(step), "improvement": 0.0}

        if torch is not None and len(rb) >= cfg.replay.batch_size:
            import numpy as np
            batch, idxs, iw = rb.sample(cfg.replay.batch_size)
            T = torch
            s = T.tensor(batch["state"], dtype=T.float32)
            a = T.tensor(batch["action"], dtype=T.int64)
            r = T.tensor(batch["reward"], dtype=T.float32)
            ns = T.tensor(batch["next_state"], dtype=T.float32)
            d = T.tensor(batch["done"], dtype=T.float32)
            w = T.tensor(iw, dtype=T.float32)

            # Compute latents and TD targets
            z = online.encode(s)
            td_tgt = double_dqn_targets(online, target, ns, r, d, hp.gamma)

            # Smooth the targets in latent space
            td_tgt_sm, diag = apply_batch_smoothing(h=td_tgt, z=z, step=step, td_var=float(td_stats["var"]), config=cfg_run)
            # Compute FEP metric for diagnostics
            try:
                from agmlib.smoothing import free_energy
                knn = compute_knn(z, cfg_run.kernel_smoothing.k)
                sigmas = adaptive_sigmas(z, knn, cfg_run)
                fep_val = free_energy(h=td_tgt_sm.detach(), delta=td_tgt.detach(), z=z.detach(), knn=knn, sigmas=sigmas, temperature=1.0)
            except Exception:
                fep_val = None

            q_pred = online(s).gather(1, a.view(-1, 1)).squeeze(1)
            loss = ((w * (q_pred - td_tgt_sm.detach()) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(online.parameters(), max_norm=hp.max_grad_norm)
            opt.step()

            # Soft target update
            tau = 0.005
            with T.no_grad():
                for p_t, p_o in zip(target.parameters(), online.parameters()):
                    p_t.data.copy_((1 - tau) * p_t.data + tau * p_o.data)

            # Update priorities from absolute TD error
            td_err = (td_tgt_sm.detach() - q_pred.detach()).abs().cpu().numpy()
            rb.update_priorities(idxs, td_err + 1e-6)

        # Optional tiny smoothing demo (torch optional)
        if args.demo_smoothing:
            try:
                import importlib
                torch = importlib.import_module("torch")
                torch.manual_seed(0)
                # Apply CLI overrides to config for demo only (shallow copy via model_copy)
                cfg_demo = cfg.model_copy(deep=True)
                if args.k is not None:
                    object.__setattr__(cfg_demo.kernel_smoothing, "k", int(args.k))
                if args.tau is not None:
                    object.__setattr__(cfg_demo.kernel_smoothing, "trust_region_tau", float(args.tau))
                if args.dim_exponent is not None:
                    object.__setattr__(cfg_demo.kernel_smoothing, "dim_exponent", float(args.dim_exponent))
                if args.td_var_min is not None:
                    object.__setattr__(cfg_demo.kernel_smoothing, "td_var_min", float(args.td_var_min))
                if args.disagreement_min is not None:
                    object.__setattr__(cfg_demo.kernel_smoothing, "disagreement_min", float(args.disagreement_min))

                batch, dim = 64, (cfg_demo.kernel_smoothing.latent_dim or 8)
                z = torch.randn(batch, dim)
                h = torch.randn(batch)
                h_updated, diag = apply_batch_smoothing(
                    h=h, z=z, step=step, td_var=float(td_stats["var"]), config=cfg_demo
                )
                # Telemetry histograms & stats
                metrics.log({
                    "step": step,
                    "sigma_hist": histogram(diag["sigmas"].squeeze(1), bins=args.bins),
                    "delta_h_hist": histogram(diag["delta_h"], bins=args.bins),
                    "avg_disagreement": diag["avg_disagreement"],
                    "gate": diag["gate"],
                    "sigma_stats": diag["sigma_stats"],
                    "delta_h_stats": diag["delta_h_stats"],
                })
            except Exception as _:
                pass

        # Example: emit adaptation applied event with placeholder decision
        decision = SmoothingDecisionEntity(
            applied=True,
            alpha_step_used=schedule_alpha_step(step, config=cfg),
            tau_clip=cfg.kernel_smoothing.trust_region_tau,
            gated_by_uncertainty=True,
            sigma_stats={"min": cfg.kernel_smoothing.sigma_min, "max": cfg.kernel_smoothing.sigma_max},
            fep_value=(fep_val if 'fep_val' in locals() else None),
        )
        emit_event(
            AdaptationAppliedEvent(
                subject_id=str(decision.ecs_id),
                step=step,
                decision={"alpha_step": decision.alpha_step_used},
                pre_metrics=td_stats,
                post_metrics=td_stats,
            )
        )

        # Early stop decision (optionally honor eval_interval)
        should_eval = True
        if args.honor_eval_interval:
            should_eval = (step % cfg_run.early_stopping.eval_interval) == 0
        if should_eval:
            stop_decision = early_stop.update(td_stats, q_stab, eval_m)
            metrics.log({"step": step, "early_stop": stop_decision.should_stop, "patience_left": stop_decision.patience_left})
            if stop_decision.should_stop:
                emit_event(EarlyStopTriggeredEvent(subject_id=str(stop_decision.ecs_id), step=step, decision={"reason": stop_decision.reason}))
                break


if __name__ == "__main__":
    main()

