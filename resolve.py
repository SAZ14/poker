"""
Subgame re-solving for Leduc's second street (the flop).

The idea, from the Libratus / DeepStack line of work: a blueprint strategy
computed for the whole game is necessarily coarse, because its iterations are
spread over every information set. Once the public state is known -- here,
once the public card is dealt and the preflop betting is over -- the remaining
game is tiny, and you can afford to solve it far more accurately than the
blueprint did. So you keep the blueprint for the early streets and re-solve
each subgame on arrival.

A Leduc flop subgame is identified by (preflop betting line, public card).
Because Leduc's information-set key is

    f"{private_card}|{public_card}|{preflop_hist}/{flop_hist}"

every flop information set belongs to exactly one subgame, so the re-solved
strategies splice into the blueprint without any conflict: the combined
strategy is the blueprint's preflop entries plus the re-solved flop entries.

What is re-solved here is the subgame rooted at the first flop decision, with
a root distribution over (player 0 card, player 1 card) given by

    P(deal) x pi_0^blueprint(preflop line | card_0) x pi_1^blueprint(preflop line | card_1)

normalized over the pairs consistent with the public card. The subgame is
solved with vanilla (full-tree) CFR+ rather than a sampled variant: it has at
most nine deals and a handful of betting nodes, so exact traversal is both
cheaper and exact.

WHY THIS IS *UNSAFE* RE-SOLVING
-------------------------------
This implementation is "unsafe" in the technical sense of Burch, Johanson &
Bowling ("Solving Imperfect Information Games Using Decomposition", AAAI
2014). It fixes the opponent's range at the subgame root to whatever the
blueprint would have brought there, and then solves the subgame against that
frozen distribution.

The opponent is under no obligation to play the blueprint. Once our flop
strategy is tuned to a particular opponent range, the opponent can change
their *preflop* strategy to arrive at this subgame with a different mix of
hands -- precisely the mix our re-solved strategy handles worst -- and we have
given up the blueprint's defense against it. Nothing in the re-solve bounds
how much value the opponent can gain by deviating earlier, so the combined
strategy can be *more* exploitable than the blueprint it came from, even
though the subgame itself is solved to a much tighter tolerance. This is not a
hypothetical: unsafe re-solving is known to lose to its own blueprint on some
games and some blueprints, and the measurement script here reports whichever
way it lands rather than assuming improvement.

Safe re-solving fixes this by replacing the subgame root with a gadget: the
opponent chooses, for each of their hands, either to enter the subgame or to
take an alternative payoff equal to the counterfactual value the blueprint
already guaranteed them there. Solving the gadget game constrains the
re-solved strategy to give the opponent no more than the blueprint did for
every hand, so the combined strategy is provably no more exploitable than the
blueprint. That gadget is not implemented yet; the root-distribution and
subgame-solver pieces below are the parts it would reuse.

Usage:
    python3 resolve.py                          # default 20k blueprint
    python3 resolve.py --blueprint-iterations 100000
    python3 resolve.py --sweep                  # several blueprint strengths
"""
import argparse
import time
from collections import Counter

import leduc
from mccfr import MCCFRTrainer, _regret_matching
from exploitability import exploitability

# Preflop histories that end the round without a fold, i.e. that reach the flop.
PREFLOP_LINES = ("cc", "rc", "crc", "rrc", "crrc")

BOARD_RANKS = tuple(range(leduc.RANKS))


# ---------------------------------------------------------------------------
# Root distribution from the blueprint
# ---------------------------------------------------------------------------

def _probs(table, key, actions):
    """Blueprint probabilities at `key`, uniform if the info set is unseen.
    Matches the fallback used by exploitability.py so the two agree."""
    strat = table.get(key)
    if strat is None:
        return [1.0 / len(actions)] * len(actions)
    return [strat.get(a, 0.0) for a in actions]


def preflop_reach(blueprint, player, card, hist):
    """
    Probability that `player`, holding `card`, plays their own actions along
    the preflop history `hist` under the blueprint. Player 0 acts first
    preflop, so the player to act at index i is i % 2.
    """
    p = 1.0
    for i, action in enumerate(hist):
        if i % 2 != player:
            continue
        prefix = hist[:i]
        actions = leduc.legal_actions_for_round_history(prefix)
        probs = _probs(blueprint, f"{card}|-|{prefix}/", actions)
        p *= probs[actions.index(action)]
        if p == 0.0:
            return 0.0
    return p


def deal_probability(c0, c1, board):
    """P(player 0 is dealt rank c0, player 1 rank c1, board rank `board`)
    dealing three cards without replacement from the six-card deck."""
    counts = Counter(leduc._full_deck())
    p = counts[c0] / 6.0
    counts[c0] -= 1
    if counts[c1] <= 0:
        return 0.0
    p *= counts[c1] / 5.0
    counts[c1] -= 1
    if counts[board] <= 0:
        return 0.0
    return p * counts[board] / 4.0


def subgame_root_distribution(blueprint, preflop_hist, board):
    """
    [(c0, c1, probability)] for the flop subgame reached by `preflop_hist`
    with public card `board`, under the blueprint's preflop play. Normalized;
    returns an empty list if the blueprint never reaches this subgame.
    """
    weights = []
    total = 0.0
    for c0 in BOARD_RANKS:
        for c1 in BOARD_RANKS:
            chance = deal_probability(c0, c1, board)
            if chance == 0.0:
                continue
            w = (chance
                 * preflop_reach(blueprint, 0, c0, preflop_hist)
                 * preflop_reach(blueprint, 1, c1, preflop_hist))
            if w > 0.0:
                weights.append((c0, c1, w))
                total += w
    if total == 0.0:
        return []
    return [(c0, c1, w / total) for c0, c1, w in weights]


def flop_root_state(c0, c1, board, preflop_hist):
    """The state at the first flop decision, reached by actually playing
    `preflop_hist` through the game so contributions and the deck stay
    consistent with leduc.py rather than being hardcoded here."""
    deck = leduc._full_deck()
    for c in (c0, c1):
        deck.remove(c)
    state = leduc.LeducState((c0, c1), None, 0, ("", ""),
                             (leduc.ANTE, leduc.ANTE), None, tuple(deck))
    for action in preflop_hist:
        state = state.next_state(action)
    assert state.current_player() == "CHANCE", preflop_hist
    return state.with_public(board)


# ---------------------------------------------------------------------------
# Vanilla CFR+ on a subgame
# ---------------------------------------------------------------------------

class _SubgameNode:
    __slots__ = ("actions", "regret", "strategy_sum", "current")

    def __init__(self, actions):
        self.actions = actions
        n = len(actions)
        self.regret = [0.0] * n
        self.strategy_sum = [0.0] * n
        self.current = [1.0 / n] * n

    def average(self):
        total = sum(self.strategy_sum)
        if total <= 0.0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [s / total for s in self.strategy_sum]


def _cfr(state, my_reach, cf_reach, player, t, nodes):
    """
    Vanilla CFR+ traversal of a flop subgame, returning the expected utility
    of `state` to `player`.

    my_reach: player's own reach probability, used to weight the average
              strategy (linear in t, as CFR+ prescribes).
    cf_reach: the counterfactual reach -- the opponent's reach times the root
              probability of this deal -- used to weight regrets.

    There are no chance nodes below a flop root in Leduc, so this handles only
    terminal and decision nodes.
    """
    if state.is_terminal():
        return state.utility(player)

    current = state.current_player()
    key = state.info_set_key()
    node = nodes.get(key)
    if node is None:
        node = _SubgameNode(state.legal_actions())
        nodes[key] = node

    strategy = node.current
    n = len(strategy)
    utils = [0.0] * n
    node_util = 0.0
    for i, a in enumerate(node.actions):
        if current == player:
            u = _cfr(state.next_state(a), my_reach * strategy[i], cf_reach, player, t, nodes)
        else:
            u = _cfr(state.next_state(a), my_reach, cf_reach * strategy[i], player, t, nodes)
        utils[i] = u
        node_util += strategy[i] * u

    if current == player:
        regret = node.regret
        ssum = node.strategy_sum
        for i in range(n):
            # CFR+: accumulate counterfactual regret, then clip at zero.
            r = regret[i] + cf_reach * (utils[i] - node_util)
            regret[i] = r if r > 0.0 else 0.0
            ssum[i] += t * my_reach * strategy[i]
    return node_util


def resolve_subgame(root_deals, preflop_hist, board, iterations=1000):
    """
    Solve one flop subgame with vanilla CFR+ and return its average strategy
    table, {info_set_key: {action: probability}}.

    root_deals: [(c0, c1, probability)] from subgame_root_distribution.
    """
    roots = [(flop_root_state(c0, c1, board, preflop_hist), p) for c0, c1, p in root_deals]
    nodes = {}
    for t in range(1, iterations + 1):
        # Freeze each node's current strategy for the whole iteration, so that
        # regret updates from one deal do not change the strategy used for the
        # next deal within the same iteration.
        for node in nodes.values():
            node.current = _regret_matching(node.regret)
        for player in (0, 1):
            for state, p in roots:
                _cfr(state, 1.0, p, player, t, nodes)
    return {key: {a: p for a, p in zip(node.actions, node.average())}
            for key, node in nodes.items()}


def resolve_all(blueprint, iterations=1000):
    """
    Re-solve every reachable flop subgame and return
    (combined_table, stats). The combined table is the blueprint with the
    re-solved flop information sets overwritten.
    """
    combined = dict(blueprint)
    stats = {"resolved": 0, "skipped": 0, "info_sets_replaced": 0, "subgames": []}
    for preflop_hist in PREFLOP_LINES:
        for board in BOARD_RANKS:
            root_deals = subgame_root_distribution(blueprint, preflop_hist, board)
            if not root_deals:
                stats["skipped"] += 1
                stats["subgames"].append((preflop_hist, board, 0, 0))
                continue
            table = resolve_subgame(root_deals, preflop_hist, board, iterations)
            combined.update(table)
            stats["resolved"] += 1
            stats["info_sets_replaced"] += len(table)
            stats["subgames"].append((preflop_hist, board, len(root_deals), len(table)))
    return combined, stats


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def train_blueprint(iterations, seed=0, variant="plus"):
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                           sample_chance=lambda s, rng: s.deal_public_sample(rng),
                           seed=seed, variant=variant)
    trainer.train(iterations)
    return trainer.average_strategy_table()


def compare(blueprint_iterations, resolve_iterations, seed=0, variant="plus", verbose=True):
    """Train a blueprint, re-solve its flop subgames, and measure both."""
    t0 = time.time()
    blueprint = train_blueprint(blueprint_iterations, seed, variant)
    train_time = time.time() - t0

    bp_expl, bp0, bp1 = exploitability(leduc.LeducState.enumerate_deals, blueprint)

    t0 = time.time()
    combined, stats = resolve_all(blueprint, resolve_iterations)
    resolve_time = time.time() - t0

    rs_expl, rs0, rs1 = exploitability(leduc.LeducState.enumerate_deals, combined)

    result = {
        "blueprint_iterations": blueprint_iterations,
        "resolve_iterations": resolve_iterations,
        "seed": seed,
        "variant": variant,
        "blueprint_exploitability": bp_expl,
        "resolved_exploitability": rs_expl,
        "blueprint_br": (bp0, bp1),
        "resolved_br": (rs0, rs1),
        "train_seconds": train_time,
        "resolve_seconds": resolve_time,
        "stats": stats,
    }
    if verbose:
        change = (rs_expl - bp_expl) / bp_expl * 100.0
        print(f"blueprint : {blueprint_iterations} iterations of {variant}, seed {seed} "
              f"({train_time:.1f}s)")
        print(f"re-solve  : {stats['resolved']} subgames, {stats['skipped']} unreachable, "
              f"{stats['info_sets_replaced']} info sets replaced, "
              f"{resolve_iterations} CFR+ iterations each ({resolve_time:.1f}s)")
        print(f"exploitability  blueprint={bp_expl:.5f}  re-solved={rs_expl:.5f}  "
              f"({change:+.1f}%)")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--blueprint-iterations", type=int, default=20_000)
    parser.add_argument("--resolve-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="plus", choices=("plain", "plus", "dcfr"))
    parser.add_argument("--sweep", action="store_true",
                        help="run several blueprint strengths and print a table")
    args = parser.parse_args(argv)

    if not args.sweep:
        compare(args.blueprint_iterations, args.resolve_iterations, args.seed, args.variant)
        return

    rows = []
    for bp_iters in (5_000, 20_000, 100_000, 500_000):
        print(f"\n--- blueprint {bp_iters} ---")
        rows.append(compare(bp_iters, args.resolve_iterations, args.seed, args.variant))

    print()
    print(f"{'blueprint iters':>15} | {'blueprint':>10} | {'re-solved':>10} | {'change':>8}")
    print("-" * 54)
    for r in rows:
        bp, rs = r["blueprint_exploitability"], r["resolved_exploitability"]
        print(f"{r['blueprint_iterations']:>15} | {bp:>10.5f} | {rs:>10.5f} | "
              f"{(rs - bp) / bp * 100:>+7.1f}%")


if __name__ == "__main__":
    main()
