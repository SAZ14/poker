"""
Benchmark the Leduc training loop: pure Python vs the native Rust backend.

    python3 bench.py                      # 1M iterations, both backends
    python3 bench.py --iterations 200000
    python3 bench.py --variant dcfr
    python3 bench.py --backends rust      # skip the slow one

Both backends run the same seed and, when both are available, the results are
compared exactly -- the native loop is meant to be bit-identical, so a
mismatch is a bug, not a tolerance to widen.
"""
import argparse
import time

import leduc
from mccfr import MCCFRTrainer, native_available


def build(backend, variant, seed):
    return MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                        sample_chance=lambda s, rng: s.deal_public_sample(rng),
                        seed=seed, variant=variant, backend=backend)


def timed(backend, iterations, variant, seed):
    trainer = build(backend, variant, seed)
    t0 = time.perf_counter()
    trainer.train(iterations)
    elapsed = time.perf_counter() - t0
    return elapsed, trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--variant", default="plus", choices=("plain", "plus", "dcfr"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", default="both", choices=("both", "python", "rust"))
    args = parser.parse_args(argv)

    if args.backends in ("both", "rust") and not native_available():
        raise SystemExit("mccfr_rs is not built. Run ./build_rust.sh first, "
                         "or pass --backends python.")

    wanted = ["python", "rust"] if args.backends == "both" else [args.backends]
    print(f"Leduc, {args.iterations:,} iterations, variant={args.variant}, seed={args.seed}")

    results = {}
    for backend in wanted:
        print(f"  running {backend} ...", flush=True)
        elapsed, trainer = timed(backend, args.iterations, args.variant, args.seed)
        results[backend] = (elapsed, trainer)
        print(f"    {elapsed:.2f}s  ({args.iterations / elapsed:,.0f} iterations/s)")

    print()
    print(f"{'backend':>8} | {'wall time':>10} | {'iterations/s':>14} | {'speedup':>8}")
    print("-" * 52)
    base = results.get("python", (None,))[0]
    for backend in wanted:
        elapsed = results[backend][0]
        speedup = f"{base / elapsed:.1f}x" if base else "-"
        print(f"{backend:>8} | {elapsed:>9.2f}s | {args.iterations / elapsed:>14,.0f} | {speedup:>8}")

    if len(results) == 2:
        a = results["python"][1].average_strategy_table()
        b = results["rust"][1].average_strategy_table()
        identical = a == b
        print()
        print(f"results bit-identical: {identical}")
        if not identical:
            raise SystemExit("backends disagree -- this is a bug")


if __name__ == "__main__":
    main()
