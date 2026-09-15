"""
Correctness tests.

1. Game-logic sanity checks (zero-sum, legal terminal/non-terminal states).
2. End-to-end MCCFR convergence checks against known theory:
   - Kuhn poker's game value to player 0 is exactly -1/18 at any Nash
     equilibrium (Kuhn, 1950), and exploitability -> 0.
   - Leduc poker has no simple closed form, so we only check that
     exploitability decreases substantially with more iterations (i.e. the
     algorithm is actually learning, not stuck/buggy).

Run with:  python3 -m pytest test_solver.py -v
"""
import random

import kuhn
import leduc
from mccfr import MCCFRTrainer, OutcomeSamplingTrainer
from exploitability import exploitability, best_response_value
from strategy import save_strategy, load_strategy, leduc_summary, parse_leduc_key
import play
import resolve


def _quick_leduc_blueprint(iterations, seed=0, variant="plus"):
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                           sample_chance=lambda s, rng: s.deal_public_sample(rng),
                           seed=seed, variant=variant)
    trainer.train(iterations)
    return trainer.average_strategy_table()


# ---------------------------------------------------------------------------
# Game logic sanity checks
# ---------------------------------------------------------------------------

def test_kuhn_zero_sum():
    rng = random.Random(0)
    for _ in range(200):
        state = kuhn.KuhnState.sample_root(rng)
        while not state.is_terminal():
            state = state.next_state(rng.choice(state.legal_actions()))
        assert state.utility(0) == -state.utility(1)


def test_kuhn_terminal_histories_match_spec():
    expected_terminal = {"pp", "bp", "bb", "pbp", "pbb"}
    reachable = set()
    for c0, c1 in [(0, 1), (1, 2), (2, 0)]:
        frontier = [kuhn.KuhnState((c0, c1), "")]
        while frontier:
            s = frontier.pop()
            if s.is_terminal():
                reachable.add(s.history)
            else:
                for a in s.legal_actions():
                    frontier.append(s.next_state(a))
    assert reachable == expected_terminal


def test_leduc_zero_sum():
    rng = random.Random(0)
    for _ in range(200):
        state = leduc.LeducState.sample_root(rng)
        while not state.is_terminal():
            player = state.current_player()
            if player == "CHANCE":
                state = state.deal_public_sample(rng)
            else:
                state = state.next_state(rng.choice(state.legal_actions()))
        assert abs(state.utility(0) + state.utility(1)) < 1e-9


def test_leduc_raise_cap_enforced():
    rng = random.Random(0)
    state = leduc.LeducState.sample_root(rng)
    # force both players to raise repeatedly; the game must cap at 2 raises/round
    while not state.is_terminal() and state.current_player() != "CHANCE":
        actions = state.legal_actions()
        if "r" in actions:
            state = state.next_state("r")
        else:
            state = state.next_state("c")
            break
    assert state._num_raises() <= leduc.RAISE_CAP


# ---------------------------------------------------------------------------
# MCCFR convergence checks
# ---------------------------------------------------------------------------

def test_kuhn_mccfr_converges_to_known_game_value():
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, seed=42)
    trainer.train(150_000)
    table = trainer.average_strategy_table()

    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, table)

    # Nash equilibrium value to player 0 is exactly -1/18 in Kuhn poker.
    assert abs(br0 - (-1 / 18)) < 0.02
    assert abs(br1 - (1 / 18)) < 0.02
    assert expl < 0.02


def test_kuhn_mccfr_exploitability_decreases_with_more_iterations():
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, seed=7)
    trainer.train(2_000)
    table_early = trainer.average_strategy_table()
    expl_early, _, _ = exploitability(kuhn.KuhnState.enumerate_deals, table_early)

    trainer.train(100_000)  # continue training the same trainer
    table_late = trainer.average_strategy_table()
    expl_late, _, _ = exploitability(kuhn.KuhnState.enumerate_deals, table_late)

    assert expl_late < expl_early


def test_leduc_mccfr_makes_progress():
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                            sample_chance=lambda s, rng: s.deal_public_sample(rng),
                            seed=3)
    trainer.train(5_000)
    table_early = trainer.average_strategy_table()
    expl_early, _, _ = exploitability(leduc.LeducState.enumerate_deals, table_early)

    trainer.train(60_000)
    table_late = trainer.average_strategy_table()
    expl_late, _, _ = exploitability(leduc.LeducState.enumerate_deals, table_late)

    assert expl_late < expl_early * 0.75  # meaningfully better, not just noise


def test_leduc_cfr_plus_beats_plain_mccfr():
    """CFR+ (regret clipping + linear averaging) should reach a lower
    exploitability than plain MCCFR after the same number of iterations
    from the same seed."""
    def make_trainer(plus):
        return MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                            sample_chance=lambda s, rng: s.deal_public_sample(rng),
                            seed=11, plus=plus)

    results = {}
    for plus in (False, True):
        trainer = make_trainer(plus)
        trainer.train(60_000)
        table = trainer.average_strategy_table()
        results[plus], _, _ = exploitability(leduc.LeducState.enumerate_deals, table)

    assert results[True] < results[False]


# ---------------------------------------------------------------------------
# Discounted CFR
# ---------------------------------------------------------------------------

def test_dcfr_lazy_discount_matches_per_iteration_product():
    """The O(1) lazy sync must equal applying every skipped iteration's
    discount one at a time."""
    import math
    from mccfr import InfoSetNode
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, variant="dcfr",
                           alpha=1.5, beta=0.0, gamma=2.0)
    for t in range(1, 8):
        trainer._push_discount(t)
    node = InfoSetNode(["p", "b"])
    node.regret_sum = [3.0, -2.0]
    node.strategy_sum = [5.0, 7.0]
    node.synced = 2          # discounts for iterations 1 and 2 already applied
    trainer._discount_node(node, 7)  # apply iterations 3..6

    pos = neg = strat = 1.0
    for s in range(3, 7):
        pos *= s ** 1.5 / (s ** 1.5 + 1)
        neg *= s ** 0.0 / (s ** 0.0 + 1)   # = 0.5 each
        strat *= (s / (s + 1)) ** 2.0
    assert math.isclose(node.regret_sum[0], 3.0 * pos, rel_tol=1e-12)
    assert math.isclose(node.regret_sum[1], -2.0 * neg, rel_tol=1e-12)
    assert math.isclose(node.strategy_sum[0], 5.0 * strat, rel_tol=1e-12)
    assert math.isclose(node.strategy_sum[1], 7.0 * strat, rel_tol=1e-12)
    assert node.synced == 6
    # a second call with nothing new to apply is a no-op
    before = list(node.regret_sum)
    trainer._discount_node(node, 7)
    assert node.regret_sum == before


def test_dcfr_kuhn_converges_to_known_game_value():
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, seed=42, variant="dcfr")
    trainer.train(150_000)
    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals,
                                    trainer.average_strategy_table())
    assert abs(br0 - (-1 / 18)) < 0.02
    assert abs(br1 - (1 / 18)) < 0.02
    assert expl < 0.02


def test_dcfr_beats_plain_on_leduc():
    def make(variant):
        return MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                            sample_chance=lambda s, rng: s.deal_public_sample(rng),
                            seed=11, variant=variant)
    results = {}
    for variant in ("plain", "dcfr"):
        trainer = make(variant)
        trainer.train(60_000)
        results[variant], _, _ = exploitability(leduc.LeducState.enumerate_deals,
                                                trainer.average_strategy_table())
    assert results["dcfr"] < results["plain"]


def test_variant_aliases_and_validation():
    import pytest
    assert MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root).variant == "plain"
    assert MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, plus=True).variant == "plus"
    assert OutcomeSamplingTrainer(sample_root=kuhn.KuhnState.sample_root,
                                  variant="dcfr").variant == "dcfr"
    with pytest.raises(ValueError):
        MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, variant="bogus")


# ---------------------------------------------------------------------------
# Outcome-sampling MCCFR
# ---------------------------------------------------------------------------

def test_outcome_sampling_kuhn_converges_to_known_game_value():
    trainer = OutcomeSamplingTrainer(sample_root=kuhn.KuhnState.sample_root, seed=42)
    trainer.train(300_000)
    table = trainer.average_strategy_table()

    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, table)

    assert abs(br0 - (-1 / 18)) < 0.03
    assert abs(br1 - (1 / 18)) < 0.03
    assert expl < 0.03


def test_outcome_sampling_visits_every_kuhn_info_set_and_improves():
    """Epsilon-greedy exploration must reach all 12 Kuhn info sets, and the
    importance-weighted updates must actually reduce exploitability."""
    trainer = OutcomeSamplingTrainer(sample_root=kuhn.KuhnState.sample_root, seed=8)
    trainer.train(3_000)
    assert len(trainer.nodes) == 12
    expl_early, _, _ = exploitability(kuhn.KuhnState.enumerate_deals,
                                      trainer.average_strategy_table())
    trainer.train(150_000)
    expl_late, _, _ = exploitability(kuhn.KuhnState.enumerate_deals,
                                     trainer.average_strategy_table())
    assert expl_late < expl_early * 0.5


def test_outcome_sampling_leduc_makes_progress():
    trainer = OutcomeSamplingTrainer(sample_root=leduc.LeducState.sample_root,
                                     sample_chance=lambda s, rng: s.deal_public_sample(rng),
                                     seed=3)
    trainer.train(10_000)
    expl_early, _, _ = exploitability(leduc.LeducState.enumerate_deals,
                                      trainer.average_strategy_table())
    trainer.train(300_000)
    expl_late, _, _ = exploitability(leduc.LeducState.enumerate_deals,
                                     trainer.average_strategy_table())
    assert len(trainer.nodes) == 288
    assert expl_late < expl_early * 0.75


def test_leduc_king_rarely_folds_preflop_facing_bet():
    """Poker sanity check on the learned strategy: holding the best card
    preflop and facing a bet, folding should be (almost) never right."""
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                            sample_chance=lambda s, rng: s.deal_public_sample(rng),
                            seed=5, plus=True)
    trainer.train(60_000)
    table = trainer.average_strategy_table()
    king = 2
    for hist in ("r", "cr", "rr", "crr"):  # every preflop spot facing a bet/raise
        probs = table[f"{king}|-|{hist}/"]
        assert probs["f"] < 0.05, (hist, probs)


# ---------------------------------------------------------------------------
# Native (Rust) backend
# ---------------------------------------------------------------------------

import pytest
from mccfr import native_available

requires_native = pytest.mark.skipif(not native_available(),
                                     reason="mccfr_rs not built (run ./build_rust.sh)")


def _leduc_trainer(backend, variant="plus", seed=3):
    return MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                        sample_chance=lambda s, rng: s.deal_public_sample(rng),
                        seed=seed, variant=variant, backend=backend)


def test_training_is_reproducible_across_machines():
    """
    Golden values: a fixed seed must produce exactly these floats on any
    IEEE-754 machine, in either backend.

    This guards a real bug that CI caught. The node utility used to be
    computed with `np.dot`, which numpy dispatches to an OpenBLAS kernel
    selected from the CPU at runtime; it fuses the multiply and add on some
    machines and not on others, and the two round differently. Two identical
    CI runners disagreed about this seed. The hot loop now uses only plain
    IEEE-754 multiply and add, which are exactly specified, so if this test
    ever fails again something has reintroduced a contracted or
    hardware-dispatched float operation.
    """
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                           sample_chance=lambda s, rng: s.deal_public_sample(rng),
                           seed=12345, variant="plus")
    trainer.train(3_000)
    table = trainer.average_strategy_table()
    assert len(trainer.nodes) == 288
    assert table["2|-|/"] == {
        "c": 0.3160416988924218, "r": 0.6839583011075783}
    assert table["0|-|r/"] == {
        "c": 0.37993138182202757, "f": 0.49243377865942783, "r": 0.12763483951854462}
    assert table["1|1|rc/cr"] == {
        "c": 0.024916924967188853, "f": 0.005556951774594398, "r": 0.9695261232582167}


@requires_native
@pytest.mark.parametrize("variant", ["plain", "plus", "dcfr"])
def test_native_backend_is_bit_identical_to_python(variant):
    """The whole point of the port: same seed, same numbers, to the last bit."""
    py = _leduc_trainer("python", variant)
    rs = _leduc_trainer("rust", variant)
    assert py.backend == "python" and rs.backend == "rust"
    py.train(8_000)
    rs.train(8_000)

    assert set(py.nodes) == set(rs.nodes)
    for key, node in py.nodes.items():
        other = rs.nodes[key]
        assert node.actions == other.actions
        assert node.regret_sum == other.regret_sum
        assert node.strategy_sum == other.strategy_sum
    assert py.average_strategy_table() == rs.average_strategy_table()
    assert py.iteration == rs.iteration
    assert py.rng.getstate() == rs.rng.getstate()   # RNG stream stays in lockstep


@requires_native
def test_native_backend_matches_across_split_train_calls():
    """Continuing training must be equivalent to one longer run, on both
    backends -- this is what CFR+'s linear weighting and DCFR's lazy discount
    depend on."""
    py = _leduc_trainer("python", "dcfr")
    rs = _leduc_trainer("rust", "dcfr")
    for chunk in (1_000, 2_500, 4_000):
        py.train(chunk)
        rs.train(chunk)
    assert py.average_strategy_table() == rs.average_strategy_table()
    assert py.iteration == rs.iteration == 7_500


@requires_native
def test_native_backend_reports_on_the_same_iterations():
    seen = {"python": [], "rust": []}
    for backend in ("python", "rust"):
        trainer = _leduc_trainer(backend)
        trainer.train(2_500, report_every=1_000,
                      on_report=lambda t, b=backend: seen[b].append(t))
    assert seen["python"] == seen["rust"] == [1_000, 2_000]


@requires_native
def test_native_backend_is_selected_automatically_for_leduc_only():
    assert _leduc_trainer("auto").backend == "rust"
    # Kuhn has no native loop
    assert MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root).backend == "python"
    # a non-standard chance hook must not be silently replaced by Leduc's
    odd = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                       sample_chance=lambda s, rng: s.with_public(s.deck_remaining[0]))
    assert odd.backend == "python"
    # outcome sampling traverses differently, so it never uses the native loop
    outcome = OutcomeSamplingTrainer(sample_root=leduc.LeducState.sample_root,
                                     sample_chance=lambda s, rng: s.deal_public_sample(rng))
    assert outcome.backend == "python"


@requires_native
def test_backend_rust_raises_when_it_cannot_be_used():
    with pytest.raises(RuntimeError):
        MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, backend="rust")


def test_backend_argument_is_validated():
    with pytest.raises(ValueError):
        MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, backend="c++")


def test_python_backend_always_works_without_the_extension():
    """The extension is optional: forcing the Python backend must produce a
    working trainer whether or not mccfr_rs was built."""
    trainer = _leduc_trainer("python")
    trainer.train(2_000)
    assert trainer.backend == "python"
    assert len(trainer.nodes) > 0
    table = trainer.average_strategy_table()
    assert all(abs(sum(p.values()) - 1.0) < 1e-9 for p in table.values())


# ---------------------------------------------------------------------------
# Subgame re-solving (Leduc flop)
# ---------------------------------------------------------------------------

def test_preflop_lines_match_the_games_round_over_set():
    assert set(resolve.PREFLOP_LINES) == set(leduc._ROUND_OVER)


def test_deal_probabilities_sum_to_one():
    total = sum(resolve.deal_probability(a, b, c)
                for a in range(3) for b in range(3) for c in range(3))
    assert abs(total - 1.0) < 1e-12
    # three of a rank is impossible with two copies of each
    assert resolve.deal_probability(0, 0, 0) == 0.0


def test_flop_root_state_carries_the_preflop_pot():
    expected = {"cc": 2, "rc": 6, "crc": 6, "rrc": 10, "crrc": 10}
    for hist, pot in expected.items():
        state = resolve.flop_root_state(0, 1, 2, hist)
        assert sum(state.contrib) == pot
        assert state.contrib[0] == state.contrib[1]   # both called
        assert state.current_player() == 0            # P0 acts first on the flop
        assert state.round_idx == 1 and state.public == 2


def test_subgame_root_distribution_normalizes_and_respects_the_deck():
    dist = resolve.subgame_root_distribution({}, "cc", 0)  # uniform blueprint
    assert abs(sum(p for _, _, p in dist) - 1.0) < 1e-12
    pairs = {(c0, c1) for c0, c1, _ in dist}
    # board holds one of the two Jacks, so both players cannot also hold one
    assert (0, 0) not in pairs
    # a pair of ranks that avoids the board card is twice as likely as one using it
    weight = {(c0, c1): p for c0, c1, p in dist}
    assert abs(weight[(1, 2)] - 2 * weight[(0, 1)]) < 1e-12


def test_preflop_reach_multiplies_only_the_players_own_actions():
    # P0 bets then calls; P1 raises. Under a blueprint that always raises/bets,
    # P0's reach for "rrc" is P(bet) * P(call) and P1's is P(raise).
    blueprint = {
        "2|-|/": {"c": 0.0, "r": 1.0},
        "2|-|r/": {"c": 0.0, "f": 0.0, "r": 1.0},
        "2|-|rr/": {"c": 0.25, "f": 0.75},
    }
    assert resolve.preflop_reach(blueprint, 0, 2, "rrc") == 0.25
    assert resolve.preflop_reach(blueprint, 1, 2, "rrc") == 1.0


def test_resolved_subgame_returns_valid_distributions():
    blueprint = _quick_leduc_blueprint(3_000)
    dist = resolve.subgame_root_distribution(blueprint, "rc", 1)
    table = resolve.resolve_subgame(dist, "rc", 1, iterations=50)
    assert table
    for key, probs in table.items():
        assert key.split("|")[1] == "1"          # right board card
        assert key.split("|")[2].split("/")[0] == "rc"   # right preflop line
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(p >= 0.0 for p in probs.values())


def test_resolve_all_replaces_every_flop_info_set_and_keeps_preflop():
    blueprint = _quick_leduc_blueprint(5_000)
    combined, stats = resolve.resolve_all(blueprint, iterations=50)
    assert stats["resolved"] == 15 and stats["skipped"] == 0   # 5 lines x 3 boards
    assert set(combined) == set(blueprint)
    preflop = [k for k in blueprint if k.split("|")[1] == "-"]
    assert len(preflop) == 18
    for key in preflop:
        assert combined[key] == blueprint[key]     # preflop untouched
    flop = [k for k in blueprint if k.split("|")[1] != "-"]
    assert len(flop) == 270
    assert stats["info_sets_replaced"] == 270


def test_unsafe_resolving_improves_a_weak_blueprint():
    blueprint = _quick_leduc_blueprint(5_000)
    before, _, _ = exploitability(leduc.LeducState.enumerate_deals, blueprint)
    combined, _ = resolve.resolve_all(blueprint, iterations=500)
    after, _, _ = exploitability(leduc.LeducState.enumerate_deals, combined)
    assert after < before * 0.75


def test_unsafe_resolving_can_hurt_a_strong_blueprint():
    """The defining failure of UNSAFE re-solving, pinned as a test: the
    re-solve is a best response to the blueprint's frozen range, so once the
    blueprint is good enough, replacing its flop play makes the whole strategy
    more exploitable rather than less. Safe re-solving is what fixes this."""
    blueprint = _quick_leduc_blueprint(100_000)
    before, _, _ = exploitability(leduc.LeducState.enumerate_deals, blueprint)
    combined, _ = resolve.resolve_all(blueprint, iterations=500)
    after, _, _ = exploitability(leduc.LeducState.enumerate_deals, combined)
    assert before < 0.08        # the blueprint really is decent
    assert after > before       # and re-solving made it worse


# ---------------------------------------------------------------------------
# Strategy persistence and summary
# ---------------------------------------------------------------------------

def test_save_load_strategy_roundtrip(tmp_path):
    trainer = MCCFRTrainer(sample_root=kuhn.KuhnState.sample_root, seed=1)
    trainer.train(2_000)
    table = trainer.average_strategy_table()
    path = tmp_path / "kuhn.json"
    save_strategy(table, path)
    assert load_strategy(path) == table


def test_leduc_summary_names_actions_by_situation():
    table = {
        "2|-|/": {"c": 0.25, "r": 0.75},           # K preflop, first to act
        "2|-|r/": {"c": 0.6, "r": 0.4, "f": 0.0},  # K preflop, facing a bet
        "0|1|cc/": {"c": 1.0, "r": 0.0},           # J on a Q board after check-check
    }
    text = leduc_summary(table)
    assert "Private card: K" in text
    assert "P0 first to act" in text
    assert "P1 facing a bet" in text
    assert "Flop, board Q" in text
    assert "after check-check:" in text
    # facing a bet: fold column populated; first to act: fold column blank
    facing = next(l for l in text.splitlines() if "P1 facing a bet" in l)
    opening = next(l for l in text.splitlines() if "P0 first to act" in l)
    assert text.index("P0 first to act") < text.index("Flop, board Q")  # preflop block comes first
    assert " 60.0%   40.0%    0.0%" in facing
    assert "25.0%   75.0%       -       -       -" in opening
    assert parse_leduc_key("2|1|rc/cr") == (2, 1, "rc", "cr")
    assert parse_leduc_key("0|-|c/") == (0, None, "c", "")


# ---------------------------------------------------------------------------
# Interactive play (scripted input)
# ---------------------------------------------------------------------------

def _scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it, "q")


def test_play_session_scripted_calls_and_folds():
    """Play scripted hands against a partially trained bot: seats alternate,
    chip accounting matches the per-hand results, and quitting stops cleanly."""
    import io
    trainer = MCCFRTrainer(sample_root=leduc.LeducState.sample_root,
                           sample_chance=lambda s, rng: s.deal_public_sample(rng),
                           seed=2, plus=True)
    trainer.train(5_000)
    table = trainer.average_strategy_table()

    out = io.StringIO()
    # always call/check: every hand reaches a terminal state without folding
    hands, total = play.play_session(table, random.Random(9), _scripted(["c"] * 200), out,
                                     max_hands=6)
    text = out.getvalue()
    assert hands == 6
    assert "You are player 0" in text and "You are player 1" in text
    assert text.count("=== Hand ") == 6
    assert "Quitting" not in text
    # the per-hand lines sum to the running total
    per_hand = [float(s.split()[2]) * (1 if "win" in s else -1)
                for s in text.splitlines() if "You win" in s or "You lose" in s]
    assert abs(sum(per_hand) - total) < 1e-9

    # illegal input is rejected, fold ends the hand, q quits
    out = io.StringIO()
    hands, total = play.play_session(table, random.Random(9), _scripted(["x", "f", "q"]), out)
    text = out.getvalue()
    assert "Not a legal action" in text or "You fold" in text
    assert "Quitting" in text
    assert hands <= 1


def test_bot_action_samples_from_table():
    rng = random.Random(0)
    root = leduc.LeducState.sample_root(rng)
    key = root.info_set_key()
    table = {key: {"c": 0.0, "r": 1.0}}
    assert all(play.bot_action(table, root, rng) == "r" for _ in range(50))
    # missing info set falls back to a legal action
    assert play.bot_action({}, root, rng) in root.legal_actions()


def test_uniform_random_strategy_is_far_from_equilibrium():
    """Sanity check on the exploitability metric itself: a uniform random
    strategy should be clearly more exploitable than a trained one."""
    empty_table = {}  # exploitability() falls back to uniform for unseen info sets
    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, empty_table)
    assert expl > 0.3
