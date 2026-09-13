"""
Leduc Hold'em: a small but non-trivial poker variant with two betting rounds
and a public board card, standard in the CFR literature (Southey et al.,
2005) as the "next step up" from Kuhn poker for validating solvers.

Rules (limit betting, 2 raises per round cap)
----------------------------------------------
- Deck: 6 cards, two copies each of {J, Q, K} (ranks 0, 1, 2).
- Each of 2 players antes 1 chip and is dealt one private card.
- Round 0 (preflop): bet size = 2. Player 0 acts first.
- After round 0 (if nobody folded): one public card is dealt face up from
  the remaining 4 cards.
- Round 1 (flop): bet size = 4. Player 0 acts first.
- Raise cap: at most 2 raises per round.
- Showdown: a player whose private card ranks-matches the public card wins
  (a "pair" beats any non-pair). Otherwise the higher private card wins.
  Chips are split (tie) only if both players hold the same private rank
  without a pair being possible, which cannot happen here since it would
  require identical private cards -- disallowed by dealing without
  replacement -- so no-pair vs no-pair is decided by rank only, and equal
  private ranks always means both are the same pair-eligible rank, handled
  by the "pair beats non-pair" rule already covering the pair holder.

Betting-round encoding
-----------------------
Each round's action history is a string over {'c', 'r', 'f'}:
  'c' = check (if no bet is outstanding) or call (if facing a bet/raise)
  'r' = bet (if no bet is outstanding) or raise (if facing a bet/raise,
        and fewer than 2 raises have occurred this round)
  'f' = fold (only legal when facing a bet/raise)
A round ends (without a fold) when a player calls, i.e. the round history
is one of "cc", "rc", "crc", "rrc", "crrc".

Info set key = f"{private_card}|{public_card_or_'-'}|{round0_hist}/{round1_hist}"
"""
import itertools

RANKS = 3          # J, Q, K
COPIES = 2         # two of each rank
BET_SIZE = [2, 4]  # round0, round1
RAISE_CAP = 2
ANTE = 1


def _full_deck():
    return [r for r in range(RANKS) for _ in range(COPIES)]


class LeducState:
    def __init__(self, cards, public, round_idx, round_hists, contrib, folded, deck_remaining):
        self.cards = cards                    # (card_p0, card_p1)
        self.public = public                  # None until dealt
        self.round_idx = round_idx            # 0 or 1
        self.round_hists = round_hists        # [hist_round0_str, hist_round1_str]
        self.contrib = contrib                # [chips_in_by_p0, chips_in_by_p1]
        self.folded = folded                  # None or player index
        self.deck_remaining = deck_remaining  # tuple of cards left (for chance dealing)

    # ---- chance / setup ----
    @staticmethod
    def sample_root(rng):
        deck = _full_deck()
        rng.shuffle(deck)
        c0, c1, *rest = deck
        return LeducState((c0, c1), None, 0, ["", ""], [ANTE, ANTE], None, tuple(rest))

    @staticmethod
    def enumerate_deals():
        """All equally-likely (root state, prob) private-card deals, for exact
        best-response computation. The remaining 4 cards form the deck the
        mid-game public-card chance node deals from."""
        deck = _full_deck()
        n = len(deck)
        total = 0
        counts = {}
        for i, j in itertools.permutations(range(n), 2):
            combo = (deck[i], deck[j])
            counts[combo] = counts.get(combo, 0) + 1
            total += 1
        full_deck = _full_deck()
        for (c0, c1), cnt in counts.items():
            remaining = list(full_deck)
            remaining.remove(c0)
            remaining.remove(c1)
            state = LeducState((c0, c1), None, 0, ["", ""], [ANTE, ANTE], None, tuple(remaining))
            yield state, cnt / total

    def _round_hist(self):
        return self.round_hists[self.round_idx]

    def _num_raises(self):
        return self._round_hist().count("r")

    def _facing_bet(self):
        h = self._round_hist()
        return len(h) > 0 and h[-1] == "r"

    def _round_over(self):
        h = self._round_hist()
        return h in ("cc", "rc", "crc", "rrc", "crrc")

    def _acting_player(self):
        # Player 0 acts first in every round; alternate afterward.
        return len(self._round_hist()) % 2

    # ---- tree structure ----
    def is_terminal(self):
        if self.folded is not None:
            return True
        if self.round_idx == 1 and self._round_over():
            return True
        return False

    def current_player(self):
        if self.is_terminal():
            return None
        if self.round_idx == 0 and self._round_over():
            return "CHANCE"  # need to deal the public card
        return self._acting_player()

    def legal_actions(self):
        assert not self.is_terminal() and self.current_player() != "CHANCE"
        h = self._round_hist()
        if h == "" or h == "c":
            return ["c", "r"]
        # facing a bet/raise
        actions = ["c", "f"]
        if self._num_raises() < RAISE_CAP:
            actions.append("r")
        return actions

    def next_state(self, action):
        assert action in self.legal_actions()
        player = self._acting_player()
        opp = 1 - player
        contrib = list(self.contrib)
        hists = list(self.round_hists)
        bet_size = BET_SIZE[self.round_idx]

        if action == "f":
            return LeducState(self.cards, self.public, self.round_idx, hists,
                               contrib, player, self.deck_remaining)

        if action == "c":
            if self._facing_bet():
                contrib[player] = contrib[opp]  # call up to opponent's contribution
            hists[self.round_idx] = hists[self.round_idx] + "c"
        elif action == "r":
            # raise/bet: put in enough to call (if facing a bet) plus one bet_size
            to_call = max(0, contrib[opp] - contrib[player])
            contrib[player] += to_call + bet_size
            hists[self.round_idx] = hists[self.round_idx] + "r"

        return LeducState(self.cards, self.public, self.round_idx, hists,
                           contrib, None, self.deck_remaining)

    def deal_public_chance_outcomes(self):
        """Used for exact best-response search: all possible public cards
        with their conditional probabilities given the deck remaining."""
        deck = self.deck_remaining
        n = len(deck)
        counts = {}
        for c in deck:
            counts[c] = counts.get(c, 0) + 1
        for card, cnt in counts.items():
            yield card, cnt / n

    def deal_public_sample(self, rng):
        card = rng.choice(self.deck_remaining)
        return self.with_public(card)

    def with_public(self, card):
        remaining = list(self.deck_remaining)
        remaining.remove(card)
        return LeducState(self.cards, card, 1, self.round_hists, self.contrib,
                           None, tuple(remaining))

    def info_set_key(self):
        player = self.current_player()
        pub = self.public if self.public is not None else "-"
        return f"{self.cards[player]}|{pub}|{self.round_hists[0]}/{self.round_hists[1]}"

    def utility(self, player):
        assert self.is_terminal()
        opp = 1 - player
        pot = self.contrib[0] + self.contrib[1]
        if self.folded is not None:
            if self.folded == player:
                return -self.contrib[player]
            else:
                return self.contrib[opp]
        # showdown
        my_card, opp_card = self.cards[player], self.cards[opp]
        my_pair = (my_card == self.public)
        opp_pair = (opp_card == self.public)
        if my_pair and not opp_pair:
            win = True
        elif opp_pair and not my_pair:
            win = False
        elif my_card != opp_card:
            win = my_card > opp_card
        else:
            # identical private ranks, neither pairs the board -> split pot
            return 0.0
        return (pot - self.contrib[player]) if win else -self.contrib[player]


GAME_NAME = "leduc_poker"
NUM_PLAYERS = 2
