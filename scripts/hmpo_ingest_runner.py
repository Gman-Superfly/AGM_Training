import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, Iterable

from agmlib.events import emit_event, AdaptationAppliedEvent, EarlyStopTriggeredEvent
from agmlib.agm.telemetry import build_telemetry_from_hmpo_record, validate_telemetry
from agmlib.agm.convergence import AdaptiveAGMController
from agmlib.agm.phase import TrainingPhaseAdaptiveAGM
from agmlib.agm.hparam import AGMHyperparameterAdapter, BaseHyperparams
from agmlib.agm.multiscale import MultiScaleAGMFramework
from agmlib.agm.detectors import AGMConvergenceDetector, AGMUncertaintyEstimator
from agmlib.agm.uncertainty import UncertaintyGatingConfig, apply_uncertainty_gating


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, required=True, help="Path to HMPO logs (JSONL or CSV)")
    p.add_argument("--format", type=str, choices=["jsonl", "csv"], default="jsonl")
    p.add_argument("--out", type=str, default="logs/controllers.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Process at most N records (0 = all)")
    p.add_argument("--summary", action="store_true", help="Print a quick summary of decisions at the end")
    p.add_argument("--summary-out", type=str, default="", help="Optional JSON file path to write summary")
    p.add_argument("--hparam-config", type=str, default="", help="Optional path to HParamAdapterConfig (yaml/json)")
    p.add_argument("--phase-config", type=str, default="", help="Optional path to PhaseDetectorConfig (yaml/json)")
    p.add_argument("--convergence-config", type=str, default="", help="Optional path to ConvergenceConfig (yaml/json)")
    p.add_argument("--uncertainty-enabled", action="store_true", help="Enable uncertainty gating of hyperparams")
    p.add_argument("--uncertainty-mode", type=str, choices=["total", "epistemic", "aleatoric"], default="total")
    p.add_argument("--uncertainty-threshold", type=float, default=0.7)
    # Live-like toggles and guardrails
    p.add_argument("--enable-phase", action="store_true", help="Enable phase selection")
    p.add_argument("--enable-hparams", action="store_true", help="Enable hyperparameter adaptation")
    p.add_argument("--dwell-steps", type=int, default=5, help="Min steps between hyperparam changes")
    return p.parse_args()


def read_records(path: Path, fmt: str) -> Iterable[Dict[str, Any]]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except Exception:
                    continue
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)


def main() -> None:
    args = parse_args()
    in_path = Path(args.logs)
    assert in_path.exists(), f"logs file not found: {in_path}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional config loaders
    def _load_cfg(path: str, cls):
        if not path:
            return None
        p = Path(path)
        assert p.exists(), f"config not found: {p}"
        text = p.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            raw = yaml.safe_load(text)
        except Exception:
            import json as _json

            raw = _json.loads(text)
        return cls(**(raw or {}))

    from agmlib.agm.convergence import ConvergenceConfig
    from agmlib.agm.phase import PhaseDetectorConfig
    from agmlib.agm.hparam import HParamAdapterConfig

    conv_cfg = _load_cfg(args.convergence_config, ConvergenceConfig)
    phase_cfg = _load_cfg(args.phase_config, PhaseDetectorConfig)
    hparam_cfg = _load_cfg(args.hparam_config, HParamAdapterConfig)
    unc_cfg = UncertaintyGatingConfig(enabled=bool(args.uncertainty_enabled), mode=args.uncertainty_mode, threshold=float(args.uncertainty_threshold))

    # Instantiate controllers (stateless or light state) with optional configs
    conv_ctrl = AdaptiveAGMController(config=conv_cfg)
    phase_ctrl = TrainingPhaseAdaptiveAGM(detector=None if phase_cfg is None else None)
    if phase_cfg is not None:
        # Recreate with custom detector
        from agmlib.agm.phase import TrainingPhaseDetector

        phase_ctrl = TrainingPhaseAdaptiveAGM(detector=TrainingPhaseDetector(config=phase_cfg))
    hparam_adapter = AGMHyperparameterAdapter(base_hyperparams=BaseHyperparams().model_dump(), config=hparam_cfg)
    multiscale = MultiScaleAGMFramework()
    conv_detector = AGMConvergenceDetector()
    uncertainty = AGMUncertaintyEstimator()

    processed = 0
    # Summary accumulators
    phase_counts: Dict[str, int] = {}
    conv_sum = 0.0
    lr_vals = []
    early_reasons: Dict[str, int] = {}
    with out_path.open("a", encoding="utf-8") as sink:
        for rec in read_records(in_path, args.format):
            # Build and validate telemetry
            tel_raw = build_telemetry_from_hmpo_record(rec)
            tel = validate_telemetry(tel_raw)

            step = int(tel["step"])
            a_hist = tel["agm"]["arithmetic_history"]
            h_hist = tel["agm"]["harmonic_history"]
            if len(a_hist) > 0 and len(h_hist) > 0:
                rate = conv_ctrl.update(a_hist, h_hist)
                spread = abs(a_hist[-1] - h_hist[-1]) / (abs(a_hist[-1]) + abs(h_hist[-1]) + 1e-8)
            else:
                rate = 0.5
                spread = 0.0

            agm_metrics = {
                "convergence_rate": rate,
                "mean_spread": spread,
                "iterations_used": max(len(a_hist), len(h_hist)),
                "timestamp": step,
            }

            multiscale.update_all_trackers(agm_metrics)
            hierarchical = multiscale.compute_hierarchical_adaptation()

            # Phase and hparam suggestions with toggles + dwell guardrail emulation
            if args.enable_phase:
                strategy, strategy_cfg = phase_ctrl.select_adaptive_strategy(
                    loss_history=[], performance_metrics=tel.get("rl", {}), agm_convergence_rate=rate
                )
            else:
                strategy, strategy_cfg = "disabled", {}
            if args.enable_hparams:
                suggestions = hparam_adapter.adapt_hyperparameters(agm_state={"agm": tel["agm"]}, performance_trend=0.0)
            else:
                suggestions = BaseHyperparams().model_dump()

            # Early stop (AGM-specific) & uncertainty
            perf_primary = {"primary_metric": float(tel["rl"]["reward"]["eval_mean"])}
            should_stop, reason = conv_detector.should_stop_training(agm_metrics=agm_metrics, performance_metrics=perf_primary)
            epi = uncertainty.estimate_epistemic_uncertainty(tel["agm"])
            alea = uncertainty.estimate_aleatoric_uncertainty(tel["agm"], performance_variance=0.0)

            # Emit events (best-effort)
            try:
                emit_event(
                    AdaptationAppliedEvent(
                        step=step,
                        decision={"phase": strategy, "hparams": suggestions},
                        pre_metrics={"conv_rate": rate, "spread": spread},
                        post_metrics={"hierarchical": hierarchical},
                    )
                )
                if should_stop:
                    emit_event(EarlyStopTriggeredEvent(step=step, decision={"reason": reason}))
            except Exception:
                pass

            # Apply uncertainty gating policy (conservative on high uncertainty)
            record_out = apply_uncertainty_gating(
                decisions={
                    "step": step,
                    "convergence_rate": rate,
                    "phase": strategy,
                    "phase_config": strategy_cfg,
                    "hyperparams": suggestions,
                    "hierarchical": hierarchical,
                    "early_stop": {"should_stop": should_stop, "reason": reason},
                    "uncertainty": {"epistemic": epi, "aleatoric": alea},
                    "toggles": {"phase": bool(args.enable_phase), "hparams": bool(args.enable_hparams)},
                },
                epistemic=epi,
                aleatoric=alea,
                base=BaseHyperparams(),
                config=unc_cfg,
            )
            sink.write(json.dumps(record_out) + "\n")

            processed += 1
            # Update summary
            phase_counts[strategy] = phase_counts.get(strategy, 0) + 1
            conv_sum += float(rate)
            if "learning_rate" in suggestions:
                lr_vals.append(float(suggestions["learning_rate"]))
            if should_stop:
                early_reasons[reason] = early_reasons.get(reason, 0) + 1
            if args.limit and processed >= args.limit:
                break

    if args.summary and processed > 0:
        # Compute a simple repo/code hash by hashing key files
        import hashlib
        import os
        h = hashlib.sha256()
        for root, _dirs, files in os.walk("src/agmlib"):
            for fn in files:
                if fn.endswith(".py"):
                    p = Path(root) / fn
                    try:
                        h.update(p.read_bytes())
                    except Exception:
                        pass
        repo_hash = h.hexdigest()[:16]
        summary = {
            "processed": processed,
            "mean_convergence_rate": conv_sum / float(processed),
            "phase_counts": phase_counts,
            "lr": {
                "count": len(lr_vals),
                "min": min(lr_vals) if lr_vals else None,
                "mean": (sum(lr_vals) / len(lr_vals)) if lr_vals else None,
                "max": max(lr_vals) if lr_vals else None,
            },
            "early_stop_reasons": early_reasons,
            "repo_hash": repo_hash,
            "configs_used": {
                "hparam_config": args.hparam_config or None,
                "phase_config": args.phase_config or None,
                "convergence_config": args.convergence_config or None,
                "uncertainty": {"enabled": bool(args.uncertainty_enabled), "mode": args.uncertainty_mode, "threshold": float(args.uncertainty_threshold)},
            },
        }
        print(json.dumps(summary))
        if args.summary_out:
            s_path = Path(args.summary_out)
            s_path.parent.mkdir(parents=True, exist_ok=True)
            s_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()


