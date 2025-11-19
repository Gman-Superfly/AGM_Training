import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="List of controllers.jsonl files to aggregate")
    p.add_argument("--out", type=str, default="logs/experiments/aggregate_summary.json")
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


def robust_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(1, len(vs) // 10)
    core = vs[k:-k] if len(vs) > 2 * k else vs
    return float(sum(core) / len(core))


def main() -> None:
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        assert p.exists(), f"input not found: {p}"

    phases: Dict[str, int] = {}
    conv_values: List[float] = []
    lr_values: List[float] = []
    early: Dict[str, int] = {}

    for p in inputs:
        for rec in read_jsonl(p):
            ph = str(rec.get("phase", "unknown"))
            phases[ph] = phases.get(ph, 0) + 1
            conv = rec.get("convergence_rate", None)
            if isinstance(conv, (int, float)):
                conv_values.append(float(conv))
            hp = rec.get("hyperparams", {})
            if isinstance(hp, dict) and "learning_rate" in hp:
                try:
                    lr_values.append(float(hp["learning_rate"]))
                except Exception:
                    pass
            es = rec.get("early_stop", {})
            if isinstance(es, dict) and es.get("should_stop", False):
                reason = str(es.get("reason", "unknown"))
                early[reason] = early.get(reason, 0) + 1

    summary = {
        "num_inputs": len(inputs),
        "phase_counts": phases,
        "mean_convergence_rate_robust": robust_mean(conv_values),
        "learning_rate": {
            "count": len(lr_values),
            "robust_mean": robust_mean(lr_values),
        },
        "early_stop_reasons": early,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()


