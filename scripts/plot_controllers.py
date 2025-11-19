import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", type=str, required=True, help="Path to controllers.jsonl")
    p.add_argument("--out-dir", type=str, default="logs/plots", help="Directory to save plots")
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


def main() -> None:
    args = parse_args()
    path = Path(args.controllers)
    assert path.exists(), f"controllers file not found: {path}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    conv = []
    lr = []
    phases = []
    for rec in read_jsonl(path):
        steps.append(int(rec.get("step", -1)))
        conv.append(float(rec.get("convergence_rate", 0.0)))
        hp = rec.get("hyperparams", {})
        if isinstance(hp, dict):
            lr.append(float(hp.get("learning_rate", 0.0)))
        else:
            lr.append(0.0)
        phases.append(str(rec.get("phase", "unknown")))

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib is required for plotting. Please `pip install matplotlib`.")
        return

    # Convergence plot
    plt.figure()
    plt.plot(steps, conv, label="convergence_rate")
    plt.xlabel("step")
    plt.ylabel("convergence_rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_rate.png", dpi=150)

    # Learning rate plot
    plt.figure()
    plt.plot(steps, lr, label="learning_rate")
    plt.xlabel("step")
    plt.ylabel("learning_rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate.png", dpi=150)

    # Phases (encode as integers for quick view)
    phase_to_id = {p: i for i, p in enumerate(sorted(set(phases)))}
    phase_ids = [phase_to_id[p] for p in phases]
    plt.figure()
    plt.plot(steps, phase_ids, ".", label="phase_id")
    plt.yticks(list(phase_to_id.values()), list(phase_to_id.keys()))
    plt.xlabel("step")
    plt.ylabel("phase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phases.png", dpi=150)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()


