import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agmlib.agm import (
    ControllerSuite,
    build_telemetry_from_hmpo_record,
    HParamAdapterConfig,
    PhaseDetectorConfig,
    ConvergenceConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, required=True, help="HMPO metrics.jsonl")
    p.add_argument("--grid", type=str, required=True, help="JSON file with parameter grid")
    p.add_argument("--out", type=str, default="logs/experiments/sweep_results.json")
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


def cartesian_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos: List[Dict[str, Any]] = []
    def backtrack(i: int, cur: Dict[str, Any]):
        if i == len(keys):
            combos.append(dict(cur))
            return
        k = keys[i]
        for v in grid[k]:
            cur[k] = v
            backtrack(i + 1, cur)
        cur.pop(k, None)
    backtrack(0, {})
    return combos


def main() -> None:
    args = parse_args()
    logs_path = Path(args.logs)
    grid_path = Path(args.grid)
    out_path = Path(args.out)
    assert logs_path.exists(), f"logs not found: {logs_path}"
    assert grid_path.exists(), f"grid not found: {grid_path}"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    combos = cartesian_product(grid)

    # Cache telemetry
    telemetry = [build_telemetry_from_hmpo_record(rec) for rec in read_jsonl(logs_path)]

    results = []
    for combo in combos:
        # Build configs (partial fields allowed)
        hp_cfg = HParamAdapterConfig(**combo.get("hparam", {})) if "hparam" in combo else HParamAdapterConfig()
        ph_cfg = PhaseDetectorConfig(**combo.get("phase", {})) if "phase" in combo else PhaseDetectorConfig()
        cv_cfg = ConvergenceConfig(**combo.get("convergence", {})) if "convergence" in combo else ConvergenceConfig()
        # Build suite with supplied configs
        suite = ControllerSuite(hparam_config=hp_cfg)
        # Simple aggregates
        conv_sum = 0.0
        conv_n = 0
        stop_count = 0
        for tel in telemetry:
            out = suite.update(tel)
            conv = float(out.get("convergence_rate", 0.0))
            conv_sum += conv
            conv_n += 1
            if out.get("early_stop", {}).get("should_stop", False):
                stop_count += 1
        results.append({
            "combo": combo,
            "mean_convergence_rate": (conv_sum / conv_n) if conv_n else None,
            "early_stop_count": stop_count,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"num_combos": len(combos), "out": str(out_path)}))


if __name__ == "__main__":
    main()


