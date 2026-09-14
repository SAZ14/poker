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

Three update variants are available on both trainers via `variant=`:

    "plain"  regret matching, uniform average (Zinkevich et al. 2007)
    "plus"   CFR+ (Tammelin 2014): cumulative regrets are clipped at zero
             after every update (regret matching+), and the average strategy
             is weighted linearly by iteration (strategy_sum += t * sigma_t).
             `plus=True` is an alias kept for backwards compatibility.
    "dcfr"   Discounted CFR (Brown & Sandholm 2019): at the end of every
             iteration t, positive cumulative regrets are multiplied by
             t^a/(t^a+1), negative ones by t^b/(t^b+1), and the accumulated
             average strategy by (t/(t+1))^g. Defaults a=1.5, b=0, g=2 are
             the paper's recommendation. With b=0 negative regret halves
             every iteration, which behaves much like CFR+'s clipping but
             keeps a little memory; g=2 weights iteration t's contribution
             to the average by roughly t^2.

DCFR's discounts are defined per iteration over ALL info sets. Applying
them eagerly would cost O(#info sets) per iteration, so they are applied
lazily: each node remembers the iteration it was last synced to, and on
its next visit the product of all skipped per-iteration factors is applied
in O(1) using running cumulative logs (regrets) and the telescoping product
prod_{s=k+1}^{t-1} (s/(s+1))^g = ((k+1)/t)^g (strategy sum). Between visits
a node's regrets do not change sign, so the positive/negative factor is
well defined. The average strategy is normalized per node, so a pending
uniform strategy-sum discount never changes the reported table.
"""
import math
import random
from array import array

import numpy as np

VARIANTS = ("plain", "plus", "dcfr")

# Optional native acceleration for Leduc. Build it with ./build_rust.sh; without
# it everything below runs in pure Python and produces the same numbers, just
# slower. See rust/src/lib.rs for what "the same numbers" means here: the native
# loop reproduces this module's RNG stream, dot-product rounding and update
# order exactly, so results are bit-identical for a given seed.
try:
    import mccfr_rs as _rs
except ImportError:  # pragma: no cover - depends on whether the build was run
    _rs = None

_VARIANT_CODES = {"plain": 0, "plus": 1, "dcfr": 2}
_PREFLOP_LINES_PROBE = ("cc", "rc", "crc", "rrc", "crrc")


def native_available():
    """True if the Rust acceleration module was built and imported."""
    return _rs is not None


def _is_standard_leduc(sample_root, sample_chance):
    """
    Whether the native Leduc loop is a faithful stand-in for these hooks.

    The root check is identity against LeducState.sample_root. The chance hook
    cannot be identity-checked (callers normally pass a fresh lambda), so it is
    probed instead: on a state from every preflop line, the supplied hook must
    deal the same public card *and* consume the same RNG draws as
    LeducState.deal_public_sample. Anything else falls back to Python.
    """
    import leduc as _leduc

    if sample_root is not _leduc.LeducState.sample_root or sample_chance is None:
        return False
    for line in _PREFLOP_LINES_PROBE:
        for seed in range(4):
            state = _leduc.LeducState.sample_root(random.Random(991 + seed))
            for action in line:
                state = state.next_state(action)
            if state.current_player() != "CHANCE":
                return False
            mine, theirs = random.Random(seed), random.Random(seed)
            try:
                got = sample_chance(state, mine)
            except Exception:
                return False
            want = state.deal_public_sample(theirs)
            if got.public != want.public or mine.getstate() != theirs.getstate():
                return False
    return True

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
    __slots__ = ("actions", "regret_sum", "strategy_sum", "synced")

    def __init__(self, actions):
        self.actions = actions
        n = len(actions)
        self.regret_sum = [0.0] * n    # cumulative counterfactual regret per action
        self.strategy_sum = [0.0] * n  # accumulated current-strategy weight per action
        self.synced = 0                # DCFR: end-of-iteration discounts applied through this iteration

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
    def __init__(self, sample_root, sample_chance=None, seed=0, plus=False,
                 variant=None, alpha=1.5, beta=0.0, gamma=2.0, backend="auto"):
        """
        sample_root:   () -> initial game state (handles the initial deal)
        sample_chance: (state, rng) -> next state, for any *mid-game* chance
                       node (e.g. Leduc's public card). Not needed for games
                       whose only randomness is the initial deal (e.g. Kuhn).
        variant:       "plain" (default), "plus" (CFR+) or "dcfr" (Discounted
                       CFR). See the module docstring.
        plus:          alias for variant="plus", kept for backwards compatibility.
        alpha, beta, gamma: DCFR discount exponents (positive regret, negative
                       regret, average strategy). Ignored unless variant="dcfr".
        backend:       "auto" (default) uses the native Leduc loop when it is
                       built and the hooks match standard Leduc, else Python.
                       "python" forces the pure-Python loop; "rust" requires the
                       native one and raises if it is unusable.
        """
        if variant is None:
            variant = "plus" if plus else "plain"
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if backend not in ("auto", "python", "rust"):
            raise ValueError(f"backend must be auto/python/rust, got {backend!r}")
        self.sample_root = sample_root
        self.sample_chance = sample_chance
        self.rng = random.Random(seed)
        self.variant = variant
        self.plus = variant == "plus"
        self._dcfr = variant == "dcfr"
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        # DCFR: cumulative log discount factors, index t = sum over iterations 1..t.
        self._cum_log_pos = array("d", [0.0])
        self._cum_log_neg = array("d", [0.0])
        self.iteration = 0  # total iterations trained so far, across train() calls
        self._nodes = {}  # info_set_key -> InfoSetNode
        self._nodes_stale = False
        self._rs = self._make_native(backend)
        self.backend = "rust" if self._rs is not None else "python"

    def _make_native(self, backend):
        """The native trainer if it may be used here, else None."""
        if backend == "python":
            return None
        # Subclasses change the traversal (see OutcomeSamplingTrainer), so the
        # native external-sampling loop is not a valid substitute for them.
        usable = (_rs is not None
                  and type(self) is MCCFRTrainer
                  and _is_standard_leduc(self.sample_root, self.sample_chance))
        if not usable:
            if backend == "rust":
                if _rs is None:
                    raise RuntimeError("backend='rust' but mccfr_rs is not built; "
                                       "run ./build_rust.sh")
                raise RuntimeError("backend='rust' only supports standard Leduc with "
                                   "MCCFRTrainer; use backend='python'")
            return None
        return _rs.LeducTrainer(_VARIANT_CODES[self.variant],
                                self.alpha, self.beta, self.gamma)

    @property
    def nodes(self):
        """info_set_key -> InfoSetNode. Materialized from the native trainer on
        demand when the native backend is in use."""
        if self._nodes_stale:
            self._sync_from_native()
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value
        self._nodes_stale = False

    def _sync_from_native(self):
        for key, actions, regret, strategy_sum, synced in self._rs.export():
            node = self._nodes.get(key)
            if node is None:
                node = InfoSetNode(list(actions))
                self._nodes[key] = node
            node.regret_sum = list(regret)
            node.strategy_sum = list(strategy_sum)
            node.synced = synced
        self._nodes_stale = False

    def _train_native(self, iterations):
        """Hand the RNG to the native loop, run, and take the stream back so
        self.rng stays usable and continuous for callers."""
        self._rs.set_rng_state(list(self.rng.getstate()[1]))
        self._rs.train(iterations)
        self.rng.setstate((3, tuple(self._rs.rng_state()), None))
        self.iteration = self._rs.iteration
        self._nodes_stale = True

    # ---- Discounted CFR support ----

    def _push_discount(self, t):
        """Record iteration t's DCFR discount factors (called once per iteration)."""
        ta = t ** self.alpha
        tb = t ** self.beta
        self._cum_log_pos.append(self._cum_log_pos[-1] + math.log(ta / (ta + 1.0)))
        self._cum_log_neg.append(self._cum_log_neg[-1] + math.log(tb / (tb + 1.0)))

    def _discount_node(self, node, t):
        """
        Apply the end-of-iteration DCFR discounts for iterations
        node.synced+1 .. t-1 to `node`, i.e. every discount that fell due
        since its last visit. Call before adding iteration t's regrets.
        """
        k = node.synced
        if k >= t - 1:
            return
        pos_f = math.exp(self._cum_log_pos[t - 1] - self._cum_log_pos[k])
        neg_f = math.exp(self._cum_log_neg[t - 1] - self._cum_log_neg[k])
        regret = node.regret_sum
        for i in range(len(regret)):
            r = regret[i]
            if r > 0.0:
                regret[i] = r * pos_f
            elif r < 0.0:
                regret[i] = r * neg_f
        sf = ((k + 1) / t) ** self.gamma
        ssum = node.strategy_sum
        for i in range(len(ssum)):
            ssum[i] *= sf
        node.synced = t - 1

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
        node = self._nodes.get(key)
        if node is None:
            node = InfoSetNode(state.legal_actions())
            self._nodes[key] = node

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
                if self._dcfr:
                    self._discount_node(node, t)
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
        if self._rs is not None:
            # Same loop, run natively. Chunked so that on_report still fires on
            # exactly the same iterations as the Python path.
            local_t, remaining = 0, iterations
            while remaining > 0:
                if report_every and on_report is not None:
                    step = min(report_every - (local_t % report_every), remaining)
                else:
                    step = remaining
                self._train_native(step)
                local_t += step
                remaining -= step
                if report_every and on_report is not None and local_t % report_every == 0:
                    on_report(local_t)
            return

        for local_t in range(1, iterations + 1):
            self.iteration += 1
            t = self.iteration
            if self._dcfr:
                self._push_discount(t)
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

    def __init__(self, sample_root, sample_chance=None, seed=0, plus=False, epsilon=0.6,
                 variant=None, alpha=1.5, beta=0.0, gamma=2.0):
        super().__init__(sample_root, sample_chance, seed, plus,
                         variant=variant, alpha=alpha, beta=beta, gamma=gamma)
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
        node = self._nodes.get(key)
        if node is None:
            node = InfoSetNode(state.legal_actions())
            self._nodes[key] = node

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
            elif self._dcfr:
                self._discount_node(node, t)
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
