import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agmlib.agm.telemetry import build_telemetry_from_hmpo_record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", type=str, required=True, help="Path to controllers.jsonl produced by hmpo_ingest_runner")
    p.add_argument("--hmpo", type=str, default="", help="Optional path to HMPO metrics.jsonl for reward analysis")
    p.add_argument("--target-reward", type=float, default=0.0, help="Reward threshold for time-to-target calculation")
    p.add_argument("--out-dir", type=str, default="", help="Output directory for experiment summary (default logs/experiments/<timestamp>)")
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def compute_time_to_target(hmpo_logs: Iterable[Dict[str, Any]], target: float) -> Optional[int]:
    if target <= 0:
        return None
    for rec in hmpo_logs:
        tel = build_telemetry_from_hmpo_record(rec)
        step = int(tel.get("step", 0))
        rew = float(tel.get("rl", {}).get("reward", {}).get("eval_mean", 0.0))
        if rew >= target:
            return step
    return None


def main() -> None:
    args = parse_args()
    ctrl_path = Path(args.controllers)
    assert ctrl_path.exists(), f"controllers file not found: {ctrl_path}"
    # Aggregate controller decisions
    phase_counts: Dict[str, int] = defaultdict(int)
    conv_sum = 0.0
    conv_n = 0
    lr_vals: List[float] = []
    early_counts: Dict[str, int] = defaultdict(int)
    steps_seen = set()
    for rec in read_jsonl(ctrl_path):
        steps_seen.add(int(rec.get("step", -1)))
        phase_counts[str(rec.get("phase", "unknown"))] += 1
        conv = rec.get("convergence_rate", None)
        if isinstance(conv, (int, float)):
            conv_sum += float(conv)
            conv_n += 1
        hp = rec.get("hyperparams", {})
        if isinstance(hp, dict) and "learning_rate" in hp:
            try:
                lr_vals.append(float(hp["learning_rate"]))
            except Exception:
                pass
        es = rec.get("early_stop", {})
        if isinstance(es, dict) and es.get("should_stop", False):
            early_counts[str(es.get("reason", "unknown"))] += 1
    # Optional time-to-target
    ttt = None
    if args.hmpo:
        hmpo_path = Path(args.hmpo)
        assert hmpo_path.exists(), f"HMPO logs not found: {hmpo_path}"
        ttt = compute_time_to_target(read_jsonl(hmpo_path), float(args.target_reward))
    summary = {
        "controllers_file": str(ctrl_path),
        "steps_processed": len(steps_seen),
        "mean_convergence_rate": (conv_sum / conv_n) if conv_n > 0 else None,
        "phase_counts": dict(phase_counts),
        "learning_rate": {
            "count": len(lr_vals),
            "min": min(lr_vals) if lr_vals else None,
            "mean": (sum(lr_vals) / len(lr_vals)) if lr_vals else None,
            "max": max(lr_vals) if lr_vals else None,
        },
        "early_stop_reasons": dict(early_counts),
        "time_to_target_reward": ttt,
        "target_reward": args.target_reward if args.target_reward > 0 else None,
    }
    # Output
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path("logs") / "experiments" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()


