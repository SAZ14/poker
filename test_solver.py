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
