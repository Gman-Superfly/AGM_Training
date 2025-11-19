import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, required=True, help="controllers.jsonl for baseline run")
    p.add_argument("--experiment", type=str, required=True, help="controllers.jsonl for controller-enabled run")
    p.add_argument("--out", type=str, default="logs/experiments/benchmark_compare.json")
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summarize(path: Path) -> Dict[str, Any]:
    n = 0
    conv_sum = 0.0
    stop_count = 0
    for rec in read_jsonl(path):
        n += 1
        conv_sum += float(rec.get("convergence_rate", 0.0))
        es = rec.get("early_stop", {})
        if isinstance(es, dict) and es.get("should_stop", False):
            stop_count += 1
    return {
        "steps": n,
        "mean_convergence_rate": (conv_sum / n) if n else None,
        "early_stop_count": stop_count,
    }


def main() -> None:
    args = parse_args()
    base = Path(args.baseline)
    exp = Path(args.experiment)
    assert base.exists(), f"baseline file not found: {base}"
    assert exp.exists(), f"experiment file not found: {exp}"

    s_base = summarize(base)
    s_exp = summarize(exp)
    diff = {
        "delta_mean_convergence_rate": (s_exp["mean_convergence_rate"] or 0.0) - (s_base["mean_convergence_rate"] or 0.0),
        "delta_early_stop_count": int(s_exp["early_stop_count"]) - int(s_base["early_stop_count"]),
    }
    out = {"baseline": s_base, "experiment": s_exp, "diff": diff}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()


