"""
Log-log plot of Leduc exploitability vs iterations for the three update
variants (plain regret matching, CFR+, Discounted CFR), external sampling,
one seed.

    python3 plots/convergence.py                      # train, save data + PNG
    python3 plots/convergence.py --iterations 100000  # shorter run
    python3 plots/convergence.py --replot             # redraw from saved JSON

Outputs plots/convergence.json (the measurements) and plots/convergence.png.
Requires matplotlib (in requirements.txt).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import leduc                                   # noqa: E402
from mccfr import MCCFRTrainer, VARIANTS       # noqa: E402
from exploitability import exploitability      # noqa: E402

LABELS = {"plain": "plain (regret matching)", "plus": "CFR+", "dcfr": "DCFR (α=1.5, β=0, γ=2)"}
DEFAULT_CHECKPOINTS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]


def measure(variant, checkpoints, seed):
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                           sample_chance=lambda s, rng: s.deal_public_sample(rng),
                           seed=seed, variant=variant)
    points = []
    done = 0
    t0 = time.time()
    for cp in checkpoints:
        trainer.train(cp - done)
        done = cp
        expl, _, _ = exploitability(leduc.LeducState.enumerate_deals,
                                    trainer.average_strategy_table())
        points.append({"iterations": cp, "exploitability": expl, "elapsed": time.time() - t0})
        print(f"  {variant:>5} iter={cp:>7} expl={expl:.5f}", flush=True)
    return points


def plot(data, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    markers = {"plain": "o", "plus": "s", "dcfr": "^"}
    for variant in VARIANTS:
        pts = data["results"].get(variant)
        if not pts:
            continue
        xs = [p["iterations"] for p in pts]
        ys = [p["exploitability"] for p in pts]
        ax.plot(xs, ys, marker=markers[variant], markersize=4, linewidth=1.5, label=LABELS[variant])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iterations")
    ax.set_ylabel("exploitability (chips per hand)")
    ax.set_title(f"Leduc Hold'em, external-sampling MCCFR, seed {data['seed']}")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path)
    print(f"wrote {png_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_CHECKPOINTS[-1],
                        help="largest checkpoint (default 500000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true", help="only redraw from the saved JSON")
    parser.add_argument("--json", default=os.path.join(HERE, "convergence.json"))
    parser.add_argument("--png", default=os.path.join(HERE, "convergence.png"))
    args = parser.parse_args(argv)

    if args.replot:
        with open(args.json) as f:
            data = json.load(f)
    else:
        checkpoints = [c for c in DEFAULT_CHECKPOINTS if c <= args.iterations]
        if not checkpoints or checkpoints[-1] != args.iterations:
            checkpoints.append(args.iterations)
        data = {"game": "leduc", "sampler": "external", "seed": args.seed,
                "checkpoints": checkpoints, "results": {}}
        for variant in VARIANTS:
            data["results"][variant] = measure(variant, checkpoints, args.seed)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)
        print(f"wrote {args.json}")

    plot(data, args.png)


if __name__ == "__main__":
    main()
