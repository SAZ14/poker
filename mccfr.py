"""
External-Sampling Monte Carlo Counterfactual Regret Minimization
(Lanctot, Waugh, Zinkevich & Bowling, "Monte Carlo Sampling for Regret
Minimization in Extensive Games", NeurIPS 2009).

Works against any game object exposing the interface used by KuhnState /
LeducState:
    current_player()      -> 0, 1, or "CHANCE" (or None if terminal)
    is_terminal()          -> bool
    legal_actions()         -> list[str]        (player nodes only)
    next_state(action)      -> new state         (player nodes only)
    info_set_key()          -> str               (player nodes only)
    utility(player)         -> float             (terminal nodes only)
    (chance nodes are handled per-game, see `sample_chance` argument)

Regret matching, average-strategy accumulation, and the external-sampling
traversal are all game-agnostic; only the tree itself (kuhn.py / leduc.py)
and how to sample a chance node are game-specific.

CFR+ (Tammelin, "Solving Large Imperfect Information Games Using CFR+",
2014) is available via `MCCFRTrainer(plus=True)`. It makes two changes:
    (a) cumulative regrets are clipped at zero after every update
        (regret matching+), so an action that has been bad for a long
        time can be re-activated quickly once it becomes good again;
    (b) the average strategy is weighted linearly by the iteration number
        (strategy_sum += t * sigma_t), which discounts the poor early
        iterates and makes the average track the current strategy faster.
"""
import random
import numpy as np


class InfoSetNode:
    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions):
        self.actions = actions
        n = len(actions)
        self.regret_sum = np.zeros(n)
        self.strategy_sum = np.zeros(n)

    def current_strategy(self):
        pos = np.maximum(self.regret_sum, 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.full(len(self.actions), 1.0 / len(self.actions))

    def average_strategy(self):
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(len(self.actions), 1.0 / len(self.actions))


class MCCFRTrainer:
    def __init__(self, sample_root, sample_chance=None, seed=0, plus=False):
        """
        sample_root:   () -> initial game state (handles the initial deal)
        sample_chance: (state, rng) -> next state, for any *mid-game* chance
                       node (e.g. Leduc's public card). Not needed for games
                       whose only randomness is the initial deal (e.g. Kuhn).
        plus:          use CFR+ updates (regret clipping at zero and
                       linear iteration weighting of the average strategy).
                       Default False gives plain external-sampling MCCFR.
        """
        self.sample_root = sample_root
        self.sample_chance = sample_chance
        self.rng = random.Random(seed)
        self.plus = plus
        self.iteration = 0  # total iterations trained so far, across train() calls
        self.nodes = {}  # info_set_key -> InfoSetNode

    def _get_node(self, state):
        key = state.info_set_key()
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(state.legal_actions())
            self.nodes[key] = node
        return node

    def _traverse(self, state, traversing_player, t):
        """
        Returns the expected utility of `state` to `traversing_player` under
        the current strategies, updating regrets and the average strategy at
        the traversing player's nodes along the way. `t` is the 1-based
        global iteration number; it is only used by CFR+ to weight the
        average-strategy accumulation.
        """
        if state.is_terminal():
            return state.utility(traversing_player)

        player = state.current_player()
        if player == "CHANCE":
            state = self.sample_chance(state, self.rng)
            return self._traverse(state, traversing_player, t)

        node = self._get_node(state)
        strategy = node.current_strategy()

        if player == traversing_player:
            action_utils = np.zeros(len(node.actions))
            for i, a in enumerate(node.actions):
                action_utils[i] = self._traverse(state.next_state(a), traversing_player, t)
            node_util = float(np.dot(strategy, action_utils))
            node.regret_sum += action_utils - node_util
            if self.plus:
                # Regret matching+: floor cumulative regrets at zero.
                np.maximum(node.regret_sum, 0.0, out=node.regret_sum)
                # Linear averaging: later iterates count more.
                node.strategy_sum += t * strategy
            else:
                node.strategy_sum += strategy
            return node_util
        else:
            a_idx = self.rng.choices(range(len(node.actions)), weights=strategy, k=1)[0]
            a = node.actions[a_idx]
            return self._traverse(state.next_state(a), traversing_player, t)

    def train(self, iterations, report_every=None, on_report=None):
        """
        Run `iterations` more iterations. The iteration counter persists
        across calls, so calling train() twice is equivalent to one longer
        call (this matters for CFR+'s linear weighting).
        """
        for local_t in range(1, iterations + 1):
            self.iteration += 1
            t = self.iteration
            root = self.sample_root(self.rng)
            for player in (0, 1):
                self._traverse(root, player, t)
            if report_every and local_t % report_every == 0 and on_report is not None:
                on_report(local_t)

    def average_strategy_table(self):
        """key -> {action: probability}"""
        table = {}
        for key, node in self.nodes.items():
            avg = node.average_strategy()
            table[key] = {a: float(p) for a, p in zip(node.actions, avg)}
        return table
