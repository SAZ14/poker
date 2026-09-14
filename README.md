# mccfr-poker

External-sampling Monte Carlo Counterfactual Regret Minimization for computing game-theory-optimal (Nash equilibrium) strategies in poker. Verified on Kuhn and Leduc Hold'em with an exact exploitability oracle.

See [EXPLANATION.md](EXPLANATION.md) for how and why it works.

## Quick start

```bash
pip install -r requirements.txt
python3 train.py kuhn  --iterations 300000
python3 train.py leduc --iterations 300000
python3 train.py leduc --iterations 300000 --plus   # CFR+ (Tammelin 2014)
python3 train.py leduc --iterations 300000 --variant dcfr  # Discounted CFR
python3 train.py kuhn  --sampler outcome            # outcome-sampling MCCFR
python3 train.py leduc --iterations 500000 --plus --save leduc.json
python3 strategy.py leduc.json                      # readable poker chart
python3 play.py --strategy leduc.json               # play heads-up against it
python3 -m pytest test_solver.py -v
```

Kuhn converges to the known Nash value of -1/18 for player 0 in a few seconds.

## Convergence

Three update rules are available on both samplers via `--variant`:

| variant | what it does |
|---|---|
| `plain` | regret matching, uniform average (Zinkevich et al. 2007) |
| `plus` | CFR+ (Tammelin 2014): clip cumulative regret at zero, weight the average linearly by iteration |
| `dcfr` | Discounted CFR (Brown & Sandholm 2019): discount positive regret by t^α/(t^α+1), negative by t^β/(t^β+1), and the average by (t/(t+1))^γ, with α=1.5, β=0, γ=2 |

Leduc Hold'em exploitability, external sampling, seed 0:

| iterations | plain | CFR+ | DCFR |
|---|---|---|---|
| 50,000 | 0.29117 | 0.08625 | 0.09204 |
| 100,000 | 0.15437 | 0.06370 | 0.06303 |
| 200,000 | 0.22763 | 0.04503 | 0.04706 |
| 500,000 | 0.14114 | 0.02914 | 0.02896 |

![Leduc convergence: exploitability vs iterations, log-log, for plain, CFR+ and DCFR](plots/convergence.png)

CFR+ and DCFR are close to each other and roughly 5x better than plain regret matching at 500k iterations. Plain is also visibly non-monotone: with uniform averaging the sampling noise of external sampling is not damped, so more iterations can temporarily make the average strategy worse. Regenerate the plot with `python3 plots/convergence.py` (add `--replot` to redraw from the saved JSON without retraining).

## Layout

| File | What it is |
|---|---|
| `mccfr.py` | Game-agnostic MCCFR trainers (external sampling, outcome sampling) and regret-matching info-set nodes |
| `kuhn.py` | Kuhn poker game tree (3 cards, one betting round) |
| `leduc.py` | Leduc Hold'em game tree (6 cards, two rounds, public card, raise cap) |
| `exploitability.py` | Exact best-response computation; the convergence metric |
| `train.py` | CLI: train, report exploitability over time, print the average strategy, `--save` it as JSON |
| `strategy.py` | Save/load strategy tables as JSON; print a Leduc strategy as a readable poker chart |
| `play.py` | Interactive CLI: play heads-up Leduc against a saved strategy, seats alternate, chips tracked |
| `plots/convergence.py` | Measures and plots exploitability vs iterations for all three variants |
| `test_solver.py` | Game-logic and convergence tests against theory |

## Adding a game

Implement a state class with `is_terminal()`, `current_player()` (returns `0`, `1`, or `"CHANCE"`), `legal_actions()`, `next_state(a)`, `info_set_key()`, `utility(player)`, plus `sample_root(rng)` and `enumerate_deals()`. If the game has mid-hand chance nodes, also provide a sampler and pass it as `sample_chance` to `MCCFRTrainer`. `mccfr.py` and `exploitability.py` need no changes.
