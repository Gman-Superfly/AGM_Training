import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--shard", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Placeholder: In a real setup, this would connect to envs and replay.
    print(f"Actor stub started with config={args.config} shard={args.shard}")


if __name__ == "__main__":
    main()

