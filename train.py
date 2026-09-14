"""
CLI: train MCCFR on Kuhn or Leduc poker and report convergence.

Usage:
    python3 train.py kuhn   --iterations 100000
    python3 train.py leduc  --iterations 200000
    python3 train.py leduc  --iterations 200000 --plus   # CFR+ updates
    python3 train.py leduc  --iterations 500000 --plus --save leduc.json
    python3 strategy.py leduc.json                        # readable chart
    python3 train.py kuhn   --sampler outcome             # outcome sampling
"""
import argparse
import time

import kuhn
import leduc
from mccfr import MCCFRTrainer, OutcomeSamplingTrainer
from exploitability import exploitability
from strategy import save_strategy

SAMPLERS = {"external": MCCFRTrainer, "outcome": OutcomeSamplingTrainer}


def _tag(game, plus, sampler):
    tag = game + ("+" if plus else "")
    return tag if sampler == "external" else f"{tag}/{sampler}"


def run_kuhn(iterations, report_every, seed, plus=False, sampler="external"):
    trainer = SAMPLERS[sampler](sample_root=kuhn.KuhnState.sample_root, seed=seed, plus=plus)
    t0 = time.time()
    tag = _tag("kuhn", plus, sampler)

    def report(t):
        table = trainer.average_strategy_table()
        expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, table)
        print(f"[{tag}] iter={t:>8}  exploitability={expl:.5f}  "
              f"(BR0={br0:.4f}, BR1={br1:.4f})  elapsed={time.time()-t0:.1f}s")

    trainer.train(iterations, report_every=report_every, on_report=report)

    table = trainer.average_strategy_table()
    print("\nFinal average strategy (info_set -> {action: prob}):")
    for key in sorted(table):
        probs = {a: round(p, 3) for a, p in table[key].items()}
        print(f"  {key:>6s}: {probs}")

    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, table)
    print(f"\nFinal exploitability: {expl:.5f} (theoretical Nash value ~ -1/18 = {-1/18:.5f} for P0)")
    return table, expl


def run_leduc(iterations, report_every, seed, plus=False, sampler="external"):
    def sample_chance(state, rng):
        return state.deal_public_sample(rng)

    trainer = SAMPLERS[sampler](sample_root=leduc.LeducState.sample_root,
                                sample_chance=sample_chance, seed=seed, plus=plus)
    t0 = time.time()
    tag = _tag("leduc", plus, sampler)

    def report(t):
        table = trainer.average_strategy_table()
        expl, br0, br1 = exploitability(leduc.LeducState.enumerate_deals, table)
        print(f"[{tag}] iter={t:>8}  exploitability={expl:.5f}  "
              f"(BR0={br0:.4f}, BR1={br1:.4f})  elapsed={time.time()-t0:.1f}s  "
              f"info_sets={len(trainer.nodes)}")

    trainer.train(iterations, report_every=report_every, on_report=report)

    table = trainer.average_strategy_table()
    expl, br0, br1 = exploitability(leduc.LeducState.enumerate_deals, table)
    print(f"\nFinal exploitability: {expl:.5f}  |  info sets discovered: {len(trainer.nodes)}")
    return table, expl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=["kuhn", "leduc"])
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plus", action="store_true",
                        help="use CFR+ (regret clipping + linear strategy averaging)")
    parser.add_argument("--sampler", choices=sorted(SAMPLERS), default="external",
                        help="external sampling (default) or outcome sampling MCCFR")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="write the final average strategy to PATH as JSON")
    args = parser.parse_args()

    report_every = args.report_every or max(1, args.iterations // 10)

    if args.game == "kuhn":
        table, _ = run_kuhn(args.iterations, report_every, args.seed,
                            plus=args.plus, sampler=args.sampler)
    else:
        table, _ = run_leduc(args.iterations, report_every, args.seed,
                             plus=args.plus, sampler=args.sampler)

    if args.save:
        save_strategy(table, args.save)
        print(f"Saved average strategy ({len(table)} info sets) to {args.save}")
