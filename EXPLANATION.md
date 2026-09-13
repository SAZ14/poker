# Monte Carlo CFR for Game-Theory-Optimal Poker

This document explains what the code in this repo does and why it works. It assumes you know what a Nash equilibrium is and can read Python, but nothing about CFR.

## 1. The problem

Poker is a two-player zero-sum game of imperfect information: you know your own cards, you don't know your opponent's, and you both act in sequence. A "game-theory-optimal" (GTO) strategy is a Nash equilibrium of this game. Against a Nash strategy no opponent can win in expectation, no matter what they do. Finding one is the whole job of a poker solver.

The natural representation is the extensive-form game tree. Every node is a history `h`: the sequence of chance events (cards dealt) and player actions so far. Because a player cannot see the opponent's card, several histories look identical to them. The set of histories a player cannot tell apart is an information set `I`. A strategy `σ` assigns, at every information set, a probability distribution over the legal actions there. It has to be the same distribution at every history inside the information set, because the player can't tell those histories apart.

In Kuhn poker (`kuhn.py`) player 0 holding a Queen at the start of the hand is one information set that contains two histories: the opponent holds a Jack, or the opponent holds a King. In Leduc (`leduc.py`) an information set is your private card, the public card if dealt, and the full betting history.

## 2. Regret, and why minimizing it finds an equilibrium

Suppose you play the game `T` times. After the fact, for each information set and each action `a`, you could ask: how much better would I have done if I had always chosen `a` here instead of following my strategy? That quantity is the cumulative regret `R^T(I, a)`. The only subtlety is the counterfactual weighting: the value of `a` at `I` is measured as if you had played to reach `I` with probability 1, weighted by the probability that chance and the opponent brought you there. This is the "counterfactual" in counterfactual regret.

The theorem that makes the whole approach work, from Zinkevich et al. (2007), has two parts. First, the sum of positive counterfactual regrets over all information sets bounds the overall regret of the whole strategy. Second, if both players' average overall regret goes to zero, then the average strategy over the `T` iterations converges to a Nash equilibrium of the zero-sum game. So the whole problem reduces to: at each information set independently, pick actions so that regret grows slower than `T`.

Regret matching does exactly that. At iteration `t`, play each action with probability proportional to its positive cumulative regret, or uniformly if no regret is positive:

```
σ^t(I, a) = max(R(I,a), 0) / Σ_b max(R(I,b), 0)
```

Regret matching guarantees the cumulative regret grows like `O(√T)`, so average regret goes to zero like `O(1/√T)`. That is `InfoSetNode.current_strategy()` in `mccfr.py`.

Two things to keep straight, because they trip everyone up the first time. The current strategy `σ^t` is what you play during training and it does not converge to anything useful on its own; it oscillates. The average strategy `σ̄^T`, the reach-weighted mean of all the `σ^t`, is the thing that converges to equilibrium. `InfoSetNode.strategy_sum` accumulates the average, and `average_strategy()` is what you actually play.

## 3. Vanilla CFR versus Monte Carlo CFR

Vanilla CFR walks the entire game tree every iteration, computing exact counterfactual values for every information set of both players. On Kuhn (12 information sets) that's trivial. On anything resembling real poker it is impossible; heads-up no-limit Texas Hold'em has on the order of 10^160 histories.

Monte Carlo CFR (Lanctot et al., 2009) replaces the full tree walk with sampling. The key insight is that you only need the regret updates to be correct in expectation. If you sample a subset of the tree and scale the observed utilities by the inverse of the probability of sampling that path, the expected regret update equals the vanilla one, and the same convergence guarantee holds, at a per-iteration cost proportional to the sampled subtree instead of the whole tree.

This repo implements external sampling, which is the most common choice for poker. On each iteration, for each player `i` in turn:

- At chance nodes, sample one outcome. Don't enumerate the deck.
- At the opponent's decision nodes, sample one action from the opponent's current strategy `σ^t`. Don't enumerate.
- At player `i`'s own decision nodes, recurse into every action, compute the value of each, and update regret and the average strategy.

The word "external" refers to everything outside player `i`'s control being sampled. The sampling probabilities cancel exactly in the regret update, so unlike outcome sampling there is no importance-weighting term to carry around. That is why `_traverse` in `mccfr.py` looks almost identical to vanilla CFR: the only difference is the `else` branch that samples one opponent action instead of looping over all of them.

The cost is variance. External sampling needs more iterations than vanilla CFR to reach the same exploitability, but each iteration is far cheaper, and the trade is enormously in its favor for large games.

## 4. Reading `mccfr.py`

`InfoSetNode` holds two vectors per information set: `regret_sum` (cumulative counterfactual regret per action) and `strategy_sum` (accumulated current-strategy probabilities, for the average).

`MCCFRTrainer._traverse(state, traversing_player)` returns the expected utility of `state` to `traversing_player`, and updates regrets along the way. The three branches:

- Terminal: return the payoff.
- Chance or opponent node: sample, recurse, return.
- Traversing player's node: compute `action_utils[a]` for each `a` by recursing, take `node_util = σ · action_utils`, add `action_utils - node_util` to `regret_sum` (that's the instantaneous counterfactual regret), add `σ` to `strategy_sum`, return `node_util`.

`train()` samples a fresh deal each iteration and runs `_traverse` once for each player. The game code is entirely decoupled: `mccfr.py` only needs `is_terminal`, `current_player`, `legal_actions`, `next_state`, `info_set_key`, and `utility`.

`MCCFRTrainer(plus=True)` switches on CFR+ (Tammelin, 2014), which changes two lines of the traversing-player branch. After the regret update, `regret_sum` is clipped at zero elementwise (regret matching+), so an action that accumulated a large negative regret can come back into play as soon as it becomes good, instead of having to climb all the way back up. And the average strategy is accumulated as `strategy_sum += t * σ` rather than `strategy_sum += σ`, so the crude early iterates are down-weighted. Both are exact in the sense that the same convergence theorem still applies, and in practice CFR+ reaches a given exploitability in far fewer iterations. `_traverse` takes the iteration number `t` as a parameter for this reason.

## 5. Knowing whether it worked: exploitability

You cannot check a poker solver by looking at the strategies, so `exploitability.py` computes the standard convergence metric. For a strategy profile `σ`, compute the best response value `BR_i(σ_{-i})`: the most player `i` could win by playing perfectly against the opponent's fixed strategy. Then

```
exploitability(σ) = ( BR_0(σ_1) + BR_1(σ_0) ) / 2
```

This is zero exactly when `σ` is a Nash equilibrium, and it measures, in chips per hand, how much a perfect opponent could win off you. It is what every CFR paper plots on the y-axis.

There is a trap in computing best responses that is worth knowing about because it silently corrupts results. A best response must choose one action per information set, not per history. If you do a naive minimax recursion and take the max at each history, you let the "best responder" act differently depending on the opponent's hidden card, which it can't see. The result is an inflated exploitability that never reaches zero even for a true equilibrium. `best_response_value` avoids this by building the explicit tree, then resolving information sets from the leaves up, pooling the reach-weighted action values from every history in the same information set before picking the single best action.

## 6. What the results look like

Kuhn poker is the sanity check because its equilibrium is known in closed form. The game value to player 0 is exactly `-1/18 ≈ -0.0556` chips per hand at every Nash equilibrium. Running `python3 train.py kuhn --iterations 300000`:

```
[kuhn] iter=   50000  exploitability=0.00192  (BR0=-0.0533, BR1=0.0571)
[kuhn] iter=  300000  exploitability=0.00177  (BR0=-0.0534, BR1=0.0570)
```

`BR0` lands on the theoretical `-1/18`, and exploitability drops from `0.33` (uniform random) to under `0.002`. The learned strategy also has the qualitative shape of the known equilibrium: player 0 never opens with a Queen, bluffs sometimes with a Jack, and opens the King at roughly three times the Jack's bluff rate, which is the `α / 3α` relationship in Kuhn's original solution.

Leduc has no closed form, so we only check that exploitability falls monotonically-ish. Over 300k iterations it goes from `0.46` to `0.12`. That is consistent with published plain external-sampling curves; getting below `0.01` needs a few million iterations or the improvements in §8.

`test_solver.py` encodes all of this as tests.

## 7. Reading the output strategy

`train.py kuhn` prints the average strategy as `info_set -> {action: prob}`. Info set keys are `<card><history>` with `0=J, 1=Q, 2=K`, `p = pass (check/fold)`, `b = bet/call`. So `2pb: {'p': 0.0, 'b': 1.0}` reads: holding a King, after I checked and the opponent bet, always call. `0: {'p': 0.75, 'b': 0.25}` reads: opening with a Jack, bluff-bet 25% of the time.

Leduc keys are `<private>|<public or ->|<round0 history>/<round1 history>` with `c = check/call`, `r = bet/raise`, `f = fold`.

## 8. Where to go from here

The code is the minimum that is correct and verifiable. The path to a real solver runs through, roughly in this order:

- CFR+ (Tammelin, 2014): clip negative regret to zero after each update and weight the average strategy linearly by iteration. Converges dramatically faster in practice and is what every modern solver uses. It's a five-line change to `InfoSetNode`.
- Linear / Discounted CFR (Brown & Sandholm, 2019): same idea, tuned discounting schedules.
- Card abstraction: in real Hold'em you bucket the ~10^6 possible hand-board combinations into a few thousand strategically similar clusters, then solve the abstract game.
- Action abstraction: discretize bet sizes (e.g. 0.5x, 1x, 2x pot, all-in). This is what commercial no-limit solvers do.
- Deep CFR (Brown et al., 2019): replace the regret tables with neural networks so you don't need explicit abstraction.
- Depth-limited solving and subgame re-solving (Libratus, Pluribus): solve the early streets offline, re-solve the later ones in real time given the actual board.

## References

- Zinkevich, Johanson, Bowling, Piccione. *Regret Minimization in Games with Incomplete Information.* NeurIPS 2007. The original CFR paper.
- Lanctot, Waugh, Zinkevich, Bowling. *Monte Carlo Sampling for Regret Minimization in Extensive Games.* NeurIPS 2009. Introduces outcome and external sampling MCCFR.
- Tammelin. *Solving Large Imperfect Information Games Using CFR+.* 2014.
- Brown, Sandholm. *Solving Imperfect-Information Games via Discounted Regret Minimization.* AAAI 2019.
- Southey et al. *Bayes' Bluff: Opponent Modelling in Poker.* UAI 2005. Defines Leduc Hold'em.
- Kuhn. *A Simplified Two-Person Poker.* 1950.
