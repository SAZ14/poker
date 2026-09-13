"""
Kuhn Poker: the canonical 3-card, 2-player poker game used to sanity-check
CFR/MCCFR implementations, because its Nash equilibrium is known in closed
form (Kuhn, 1950).

Rules
-----
- Deck: {J, Q, K} (ranks 0, 1, 2). Each of the 2 players antes 1 chip and is
  dealt one private card.
- Player 0 acts first. Each player may 'p' (pass = check/fold) or 'b' (bet
  1 chip / call).
- Histories (from the root, player 0 to act first):
    ""     -> P0: p or b
    "p"    -> P1: p or b
    "b"    -> P1: p or b
    "pp"   -> terminal, showdown, pot = 2
    "pb"   -> P0: p or b
    "bp"   -> terminal, P1 folded, P0 wins pot = 2 (net +1/-1)
    "bb"   -> terminal, showdown, pot = 4
    "pbp"  -> terminal, P0 folded, P1 wins pot = 2 (net -1/+1)
    "pbb"  -> terminal, showdown, pot = 4

Info set key = str(private_card) + history, e.g. "1pb" = Queen, history "pb".
"""
import itertools
import random

CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
ANTE = 1
BET = 1


class KuhnState:
    """A node in the Kuhn poker game tree."""

    def __init__(self, cards, history=""):
        self.cards = cards  # (card_for_p0, card_for_p1)
        self.history = history

    # ---- chance ----
    @staticmethod
    def sample_root(rng):
        cards = list(range(3))
        rng.shuffle(cards)
        return KuhnState((cards[0], cards[1]), "")

    @staticmethod
    def enumerate_deals():
        """All equally-likely (root state, prob) deals, for exact best-response
        computation."""
        for c0, c1 in itertools.permutations(range(3), 2):
            yield KuhnState((c0, c1), ""), 1.0 / 6.0

    # ---- tree structure ----
    def current_player(self):
        h = self.history
        if h in ("", "p", "b", "pb"):
            return len(h) % 2 if h != "pb" else 0
        return None  # terminal

    def is_terminal(self):
        return self.history in ("pp", "bp", "bb", "pbp", "pbb")

    def legal_actions(self):
        return ["p", "b"]

    def next_state(self, action):
        return KuhnState(self.cards, self.history + action)

    def info_set_key(self):
        player = self.current_player()
        return f"{self.cards[player]}{self.history}"

    def utility(self, player):
        """Net chip payoff to `player` at a terminal node."""
        assert self.is_terminal()
        h = self.history
        opp = 1 - player
        my_card, opp_card = self.cards[player], self.cards[opp]

        if h == "pp":
            pot_each = ANTE
            return pot_each if my_card > opp_card else -pot_each
        if h == "bp":
            # bettor's opponent folded -> bettor (player 0) wins the ante
            return ANTE if player == 0 else -ANTE
        if h == "pbp":
            # player 0 checked, player 1 bet, player 0 folded -> player 1 wins ante
            return -ANTE if player == 0 else ANTE
        if h in ("bb", "pbb"):
            pot_each = ANTE + BET
            return pot_each if my_card > opp_card else -pot_each
        raise ValueError(f"not terminal: {h}")


GAME_NAME = "kuhn_poker"
NUM_PLAYERS = 2
