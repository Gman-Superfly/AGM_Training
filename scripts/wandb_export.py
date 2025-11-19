import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", type=str, required=True, help="Path to controllers.jsonl")
    p.add_argument("--project", type=str, default="agm-training")
    p.add_argument("--run-name", type=str, default="controllers-export")
    return p.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    args = parse_args()
    path = Path(args.controllers)
    assert path.exists(), f"controllers file not found: {path}"
    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not installed. `pip install wandb` to enable export.")
        return
    run = wandb.init(project=args.project, name=args.run_name)
    for rec in read_jsonl(path):
        step = int(rec.get("step", 0))
        wandb.log(
            {
                "convergence_rate": rec.get("convergence_rate", 0.0),
                "phase": str(rec.get("phase", "unknown")),
                "lr": float(rec.get("hyperparams", {}).get("learning_rate", 0.0))
            },
            step=step,
        )
    run.finish()


if __name__ == "__main__":
    main()


