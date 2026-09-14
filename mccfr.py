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

Two samplers are provided:
    MCCFRTrainer            external sampling (Section 4.2 of Lanctot et al.)
    OutcomeSamplingTrainer  outcome sampling (Section 4.1), one trajectory per
                            iteration with importance-weighted updates

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

# Performance note
# ----------------
# Info sets here have 2-3 actions, so per-node numpy calls cost far more in
# call overhead than they save in arithmetic. The hot path in `_traverse`
# therefore uses plain Python lists and floats. The one exception is the
# dot product for the node utility: on this platform numpy routes it to an
# OpenBLAS kernel that uses fused multiply-add, whose rounding a plain
# Python `s*u` sum cannot reproduce. To keep training output bit-identical
# for a fixed seed, that single call stays as `np.dot`.
_dot = np.dot


def _regret_matching(regret_sum):
    """Positive-regret-proportional strategy as a Python list. Matches
    np.maximum(r, 0) / sum elementwise (sequential sum, same as numpy's
    for fewer than 8 elements)."""
    pos = [r if r >= 0.0 else 0.0 for r in regret_sum]
    total = 0.0
    for p in pos:
        total += p
    if total > 0.0:
        return [p / total for p in pos]
    n = len(pos)
    return [1.0 / n] * n


class InfoSetNode:
    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions):
        self.actions = actions
        n = len(actions)
        self.regret_sum = [0.0] * n    # cumulative counterfactual regret per action
        self.strategy_sum = [0.0] * n  # accumulated current-strategy weight per action

    def current_strategy(self):
        return np.array(_regret_matching(self.regret_sum))

    def average_strategy(self):
        total = 0.0
        for s in self.strategy_sum:
            total += s
        if total > 0.0:
            return np.array([s / total for s in self.strategy_sum])
        n = len(self.actions)
        return np.full(n, 1.0 / n)


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

        key = state.info_set_key()
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(state.legal_actions())
            self.nodes[key] = node

        regret = node.regret_sum
        strategy = _regret_matching(regret)
        n = len(strategy)

        if player == traversing_player:
            action_utils = [self._traverse(state.next_state(a), traversing_player, t)
                            for a in node.actions]
            node_util = float(_dot(strategy, action_utils))
            ssum = node.strategy_sum
            if self.plus:
                for i in range(n):
                    # Regret matching+: floor cumulative regrets at zero.
                    r = regret[i] + (action_utils[i] - node_util)
                    regret[i] = r if r >= 0.0 else 0.0
                    # Linear averaging: later iterates count more.
                    ssum[i] += t * strategy[i]
            else:
                for i in range(n):
                    regret[i] += action_utils[i] - node_util
                    ssum[i] += strategy[i]
            return node_util
        else:
            # Sample one opponent action. This reproduces
            # random.choices(range(n), weights=strategy, k=1)[0] exactly:
            # one rng.random() draw scaled by the cumulative total, then a
            # bisect_right over the cumulative weights capped at n - 1.
            acc = strategy[0]
            for i in range(1, n):
                acc += strategy[i]
            r = self.rng.random() * (acc + 0.0)
            hi = n - 1
            a_idx = 0
            acc = strategy[0]
            while a_idx < hi and acc <= r:
                a_idx += 1
                acc += strategy[a_idx]
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
            self._iterate(root, t)
            if report_every and local_t % report_every == 0 and on_report is not None:
                on_report(local_t)

    def _iterate(self, root, t):
        """One iteration from a freshly dealt root: a traversal per player."""
        for player in (0, 1):
            self._traverse(root, player, t)

    def average_strategy_table(self):
        """key -> {action: probability}"""
        table = {}
        for key, node in self.nodes.items():
            avg = node.average_strategy()
            table[key] = {a: float(p) for a, p in zip(node.actions, avg)}
        return table


class OutcomeSamplingTrainer(MCCFRTrainer):
    """
    Outcome-sampling MCCFR (Lanctot et al. 2009, Section 4.1).

    Each iteration samples ONE terminal history per player instead of
    expanding every action at the traversing player's nodes. Actions are
    drawn from a sampling policy sigma': the traversing player uses an
    epsilon-greedy mixture, sigma'(I,a) = eps/|A(I)| + (1-eps) sigma(I,a),
    so every action keeps positive probability (needed for the estimator to
    be unbiased); the opponent and chance are sampled on-policy.

    Because only one trajectory is seen, the counterfactual values must be
    importance-corrected by the probability of having sampled it. With z the
    sampled terminal history, h the prefix in info set I (player i to act),
    and q(.) the sampling probability under sigma',

        v~(I, a) = u_i(z) * pi_{-i}(h) * pi^sigma(ha -> z) / q(z)   if a is on z
                 = 0                                              otherwise
        v~(I)    = sigma(I, a_sampled) * v~(I, a_sampled)

    and the regret update is r(I,a) += v~(I,a) - v~(I). The average strategy
    is accumulated as pi_i(h) sigma(I,a) / q(h) so that it too is unbiased.
    Chance probabilities appear in both pi_{-i} and q and cancel, so they are
    omitted from both. This is the same estimator as OpenSpiel's
    outcome-sampling solver with a zero baseline.

    Convergence is O(1/sqrt(T)) like external sampling, but with much
    higher variance per iteration; each iteration is far cheaper, though.
    """

    def __init__(self, sample_root, sample_chance=None, seed=0, plus=False, epsilon=0.6):
        super().__init__(sample_root, sample_chance, seed, plus)
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        self.epsilon = epsilon

    def _iterate(self, root, t):
        for player in (0, 1):
            self._sample_episode(root, player, t, 1.0, 1.0, 1.0)

    def _sample_episode(self, state, traversing_player, t, my_reach, opp_reach, sample_reach):
        """
        Walk one sampled trajectory from `state` to a terminal.

        my_reach:     pi_i(h)   traversing player's own reach under sigma
        opp_reach:    pi_{-i}(h) opponent's reach under sigma (chance omitted)
        sample_reach: q(h)      probability of having sampled the path to h

        Returns u_i(z) * pi^sigma(h -> z) / q(h -> z): the sampled utility
        of `state`, importance-corrected for the tail of the trajectory below
        it. Regrets and average strategy are updated at the traversing
        player's nodes on the way back up.
        """
        if state.is_terminal():
            return state.utility(traversing_player)

        player = state.current_player()
        if player == "CHANCE":
            state = self.sample_chance(state, self.rng)
            return self._sample_episode(state, traversing_player, t,
                                        my_reach, opp_reach, sample_reach)

        key = state.info_set_key()
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(state.legal_actions())
            self.nodes[key] = node

        regret = node.regret_sum
        strategy = _regret_matching(regret)
        n = len(strategy)

        # Sampling policy: epsilon-greedy for the traversing player, on-policy otherwise.
        if player == traversing_player:
            eps = self.epsilon
            uniform = eps / n
            sample_policy = [uniform + (1.0 - eps) * p for p in strategy]
        else:
            sample_policy = strategy

        # Draw one action from the sampling policy.
        r = self.rng.random()
        acc = 0.0
        a_idx = n - 1
        for i in range(n):
            acc += sample_policy[i]
            if r < acc:
                a_idx = i
                break
        a = node.actions[a_idx]
        sigma_a = strategy[a_idx]
        q_a = sample_policy[a_idx]

        if player == traversing_player:
            child_value = self._sample_episode(state.next_state(a), traversing_player, t,
                                               my_reach * sigma_a, opp_reach, sample_reach * q_a)
            # Importance-corrected sampled counterfactual value of the taken
            # action; every other action's sampled value is zero.
            cf_action_value = child_value / q_a * opp_reach / sample_reach
            cf_value = sigma_a * cf_action_value
            ssum = node.strategy_sum
            avg_weight = my_reach / sample_reach
            if self.plus:
                avg_weight *= t
            for i in range(n):
                delta = (cf_action_value if i == a_idx else 0.0) - cf_value
                if self.plus:
                    v = regret[i] + delta
                    regret[i] = v if v >= 0.0 else 0.0
                else:
                    regret[i] += delta
                ssum[i] += avg_weight * strategy[i]
        else:
            child_value = self._sample_episode(state.next_state(a), traversing_player, t,
                                               my_reach, opp_reach * sigma_a, sample_reach * q_a)

        # Estimated value of this state: sigma(a) * child / sigma'(a), which
        # telescopes to u_i(z) pi^sigma(h -> z) / q(h -> z) up the trajectory.
        return sigma_a * child_value / q_a
