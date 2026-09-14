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
from mccfr import MCCFRTrainer
from exploitability import exploitability, best_response_value
from strategy import save_strategy, load_strategy, leduc_summary, parse_leduc_key


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


def test_uniform_random_strategy_is_far_from_equilibrium():
    """Sanity check on the exploitability metric itself: a uniform random
    strategy should be clearly more exploitable than a trained one."""
    empty_table = {}  # exploitability() falls back to uniform for unseen info sets
    expl, br0, br1 = exploitability(kuhn.KuhnState.enumerate_deals, empty_table)
    assert expl > 0.3
