import argparse
import json
import time


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--block-size", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import torch as T
    except Exception:
        print("torch required for profiling")
        return
    from agmlib.smoothing import compute_knn, adaptive_sigmas, kernel_consensus

    T.manual_seed(0)
    z = T.randn(args.batch, args.dim)
    h = T.randn(args.batch)
    # kNN
    t0 = time.perf_counter()
    knn = compute_knn(z, args.k, block_size=(None if args.block_size <= 0 else int(args.block_size)))
    t1 = time.perf_counter()
    # sigmas
    sigmas = adaptive_sigmas(z, knn, __import__("agmlib").agmlib.config.AGMConfigEntity(replay={}, kernel_smoothing={"k": args.k}, early_stopping={}, distributed={}))
    t2 = time.perf_counter()
    # consensus
    G = kernel_consensus(h, z, knn, sigmas)
    t3 = time.perf_counter()
    out = {
        "batch": args.batch,
        "dim": args.dim,
        "k": args.k,
        "block_size": (None if args.block_size <= 0 else int(args.block_size)),
        "timings_sec": {"knn": t1 - t0, "sigmas": t2 - t1, "consensus": t3 - t2},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()


