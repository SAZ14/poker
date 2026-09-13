"""
CLI: train MCCFR on Kuhn or Leduc poker and report convergence.

Usage:
    python3 train.py kuhn   --iterations 100000
    python3 train.py leduc  --iterations 200000
    python3 train.py leduc  --iterations 200000 --plus   # CFR+ updates
"""
import argparse
import time

import kuhn
import leduc
from mccfr import MCCFRTrainer
from exploitability import exploitability


def run_kuhn(iterations, report_every, seed, plus=False):
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, seed=seed, plus=plus)
    t0 = time.time()
    tag = "kuhn+" if plus else "kuhn"

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


def run_leduc(iterations, report_every, seed, plus=False):
    def sample_chance(state, rng):
        return state.deal_public_sample(rng)

    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                            sample_chance=sample_chance, seed=seed, plus=plus)
    t0 = time.time()
    tag = "leduc+" if plus else "leduc"

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
    args = parser.parse_args()

    report_every = args.report_every or max(1, args.iterations // 10)

    if args.game == "kuhn":
        run_kuhn(args.iterations, report_every, args.seed, plus=args.plus)
    else:
        run_leduc(args.iterations, report_every, args.seed, plus=args.plus)
