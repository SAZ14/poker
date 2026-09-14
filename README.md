# mccfr-poker

External-sampling Monte Carlo Counterfactual Regret Minimization for computing game-theory-optimal (Nash equilibrium) strategies in poker. Verified on Kuhn and Leduc Hold'em with an exact exploitability oracle.

See [EXPLANATION.md](EXPLANATION.md) for how and why it works.

## Quick start

```bash
pip install -r requirements.txt
python3 train.py kuhn  --iterations 300000
python3 train.py leduc --iterations 300000
python3 train.py leduc --iterations 300000 --plus   # CFR+ (Tammelin 2014)
python3 train.py leduc --iterations 500000 --plus --save leduc.json
python3 strategy.py leduc.json                      # readable poker chart
python3 -m pytest test_solver.py -v
```

Kuhn converges to the known Nash value of -1/18 for player 0 in a few seconds.

## Layout

| File | What it is |
|---|---|
| `mccfr.py` | Game-agnostic external-sampling MCCFR trainer and regret-matching info-set nodes |
| `kuhn.py` | Kuhn poker game tree (3 cards, one betting round) |
| `leduc.py` | Leduc Hold'em game tree (6 cards, two rounds, public card, raise cap) |
| `exploitability.py` | Exact best-response computation; the convergence metric |
| `train.py` | CLI: train, report exploitability over time, print the average strategy, `--save` it as JSON |
| `strategy.py` | Save/load strategy tables as JSON; print a Leduc strategy as a readable poker chart |
| `test_solver.py` | Game-logic and convergence tests against theory |

## Adding a game

Implement a state class with `is_terminal()`, `current_player()` (returns `0`, `1`, or `"CHANCE"`), `legal_actions()`, `next_state(a)`, `info_set_key()`, `utility(player)`, plus `sample_root(rng)` and `enumerate_deals()`. If the game has mid-hand chance nodes, also provide a sampler and pass it as `sample_chance` to `MCCFRTrainer`. `mccfr.py` and `exploitability.py` need no changes.
