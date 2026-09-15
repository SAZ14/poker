# Plan: heads-up limit Texas Hold'em with pluggable card abstraction

Status: awaiting approval. Nothing below is implemented yet.

## 1. Goal

Add `holdem.py`, a heads-up **limit** Texas Hold'em game that exposes the same
state interface as `leduc.py`, so `MCCFRTrainer` and `OutcomeSamplingTrainer`
train on it unchanged. The game's `info_set_key()` routes the player's cards
through a pluggable `abstract(private_cards, board) -> bucket` hook. Ship one
abstraction: hand-strength percentile via Monte Carlo equity, 10 buckets per
street. Exact exploitability is intractable here, so add
`evaluate_vs_random.py` as a proxy metric: average chips per hand against a
uniform-random opponent over 20k hands.

## 2. Rules (fixed-limit, heads-up)

| item | value |
|---|---|
| deck | standard 52 cards (`treys` integer cards) |
| private cards | 2 per player |
| board | flop 3, turn 1, river 1 |
| blinds | player 0 is the dealer / small blind (posts 1), player 1 is the big blind (posts 2) |
| bet size | 2 on preflop and flop (small bet), 4 on turn and river (big bet) |
| order of action | preflop: dealer (player 0) acts first; flop, turn, river: big blind (player 1) acts first |
| raise cap | 4 bets per street (a bet plus 3 raises), the common limit rule; a module constant like Leduc's `RAISE_CAP` |
| showdown | best 5 of 7 via `treys.Evaluator`; ties split the pot |

Action encoding is the same alphabet as Leduc so `strategy.py`'s action
names carry over: `c` = check/call, `r` = bet/raise, `f` = fold (only when
facing a bet). Preflop the small blind is "facing a bet" of 1 chip, so its
first decision is call/raise/fold, and the big blind may check if the small
blind only called. A street ends when a player calls after a bet, or both
players check; the river ending without a fold is a showdown.

Chips: `utility(player)` returns net chips exactly like Leduc (winner gets the
loser's contribution, folder loses their contribution, split returns 0).

## 3. State interface

`HoldemState` implements exactly what `README.md` "Adding a game" lists:
`is_terminal()`, `current_player()` (0, 1, or `"CHANCE"`), `legal_actions()`,
`next_state(a)`, `info_set_key()`, `utility(player)`. Internals follow the
optimized `leduc.py` pattern: immutable states, per-street history strings in
a tuple, contributions in a tuple, acting player and terminal status computed
once in `__init__`, `legal_actions()` from a lookup table keyed by the current
street's history.

Two differences from Leduc, both deliberate:

- **Cards are dealt once at the root.** `sample_root(rng)` draws all nine
  cards (2 + 2 + 5) with the caller's `rng`. Street transitions are still
  `"CHANCE"` nodes and `sample_chance(state, rng)` reveals the next street
  from the pre-dealt board. Statistically identical to dealing at each street
  (chance is independent of actions), cheaper, and it keeps chance outside
  the info set until revealed.
- **`enumerate_deals()` and `deal_public_chance_outcomes()` raise
  `NotImplementedError`** with a message pointing at `evaluate_vs_random.py`.
  Exact best response over 52-card deals is not something this repo will
  attempt, and a silent stub would be worse than a loud one.

Because the abstraction is pluggable, the game is built as an object rather
than a bare class:

```python
game = holdem.HoldemGame(abstract=holdem.EquityBucketAbstraction())
trainer = MCCFRTrainer(sample_root=game.sample_root,
                       sample_chance=game.sample_chance, seed=0)
```

Every `HoldemState` holds a reference to its `HoldemGame`, and
`info_set_key()` calls `game.abstract(private_cards, board)`.

## 4. Info set key and recall

```
"{street}:{bucket}|{preflop_hist}/{flop_hist}/{turn_hist}/{river_hist}"
```

Betting history is perfect recall; card information is **imperfect recall**:
the key carries only the current street's bucket, not the sequence of buckets
from earlier streets. This is the standard trade-off in abstracted poker bots
and is what keeps the game small enough to train in pure Python:

| recall over buckets | rough info-set count with 10 buckets |
|---|---|
| imperfect (proposed) | 10 buckets x betting decision points, on the order of 10^4 to 10^5 |
| perfect | up to 10^4 bucket sequences x the same, on the order of 10^7 to 10^8 |

Perfect recall is a one-line change to the key format; I will note where and
leave it off by default. Decision: I recommend imperfect recall. Say so if you
want perfect recall instead.

## 5. Abstraction hook

```python
def abstract(private_cards, board) -> int      # bucket id in [0, n_buckets)
```

`private_cards` is a 2-tuple and `board` a tuple of 0, 3, 4 or 5 `treys`
cards. The street is implied by `len(board)`. Two implementations:

- **`EquityBucketAbstraction(n_buckets=10, samples=100, seed=0)`**, the
  requested one. Equity of a hand on a board is the Monte Carlo estimate of
  P(win) + 0.5 P(tie) against one uniformly random opponent hand, rolling out
  the rest of the board uniformly. Buckets are **percentiles**, not equal
  equity widths: at build time the abstraction samples random situations per
  street, computes their equities, and stores the 9 decile thresholds per
  street, so each bucket holds about a tenth of hands. `abstract` computes the
  hand's equity and returns how many thresholds it exceeds.
  - Preflop is precomputed exactly over the 169 canonical hand classes
    (pairs, suited, offsuit) with a larger sample count, since every hand
    lands in one of those classes.
  - Postflop equities are memoized in a dict keyed by the sorted cards, so
    repeat visits to the same hand and board are free. No suit isomorphism in
    this pass; it is a follow-up that would raise the hit rate.
  - The thresholds and preflop table are written to `holdem_abstraction.json`
    on first build (about 10-20 seconds) and loaded afterwards. The file is
    committed so training and evaluation are reproducible and CI does not
    rebuild it. A `--rebuild` path regenerates it.
- **`constant_abstraction`** returns 0 for every hand. It gives a game with
  only betting information, useful for fast tests of the game tree and the
  trainers, independent of `treys` speed.

## 6. Cost estimate and what to expect

The equity hook dominates training cost. With 100 rollouts per uncached call
and `treys` at a few microseconds per 7-card evaluation, one uncached
`info_set_key()` costs about half a millisecond. External sampling touches a
few dozen info sets per iteration, so expect roughly 10-30 ms per iteration
before caching helps: about 100k iterations in 20-40 minutes on this machine.
Outcome sampling is several times cheaper per iteration. `train.py` will
print info-set count, elapsed time and iterations per second instead of
exploitability for this game. I will report measured numbers with the
implementation rather than promise these.

## 7. `evaluate_vs_random.py`

```
python3 evaluate_vs_random.py --strategy holdem.json [--hands 20000] [--seed 0]
```

- Loads a saved strategy table and the same abstraction, deals hands with
  the game's `sample_root`, and plays the bot (sampling from its average
  strategy at its info set, uniform fallback for unseen info sets) against an
  opponent that picks uniformly among legal actions.
- Seats alternate every hand so blind position is balanced.
- Reports mean chips per hand with a standard error and a 95% interval, plus
  hands played and the fraction of info sets that were missing from the
  table. Optional `--opponent {random,call}`: an always-call opponent is a
  second cheap baseline that is harder to beat than random and worth having
  for the same price. Default stays `random` as requested.

## 8. Integration

- `requirements.txt`: add `treys>=0.1.8` (pure Python, no other deps).
- `train.py`: add `holdem` to the game choices, no exploitability reports,
  `--save` works as before, `--sampler` and `--plus` apply.
- `strategy.py`: `save_strategy` / `load_strategy` already work on any table.
  The Leduc chart stays Leduc-only; a preflop chart over the 169 classes is a
  natural follow-up but not part of this change.
- `README.md` layout table and quick start, `EXPLANATION.md` section 8 gets a
  short paragraph on what the abstraction does and does not buy.

## 9. Tests (added to `test_solver.py`)

Game logic, all fast, using `constant_abstraction` where cards do not matter:

- zero-sum over random playouts, correct blind postings and first-to-act per
  street, raise cap enforced, street transitions in order, showdown pays the
  `treys`-best hand and splits ties (hand-picked boards).
- betting-sequence enumeration: count of distinct per-street histories and
  that every legal history is in the lookup table.

Abstraction:

- percentile thresholds are sorted and strictly increasing per street;
  preflop table has 169 entries; pocket aces bucket 9, seven-deuce offsuit
  bucket 0; on a board where the hand holds the nuts the river bucket is 9.
- `abstract` is deterministic across calls (memoization does not change
  results) and returns ints in range.

End to end, kept small for CI:

- `MCCFRTrainer` and `OutcomeSamplingTrainer` train a few thousand iterations
  on the constant-abstraction game and on the equity game without error and
  discover info sets.
- `evaluate_vs_random` on a lightly trained bot beats the random opponent by
  a positive margin over a few thousand hands, with a fixed seed. Random play
  is very weak, so this holds well before convergence; I will pick the
  iteration count and margin from measurements so it is not flaky.

The full suite should stay under about two minutes on CI.

## 10. Out of scope, and why

- Exact exploitability or best response for Hold'em: intractable, by design
  of this task.
- Suit-isomorphism canonicalization, bucket-sequence (perfect recall)
  abstraction, potential-aware or k-means bucketing, action abstraction,
  no-limit betting. Each is a follow-up that fits behind the same hook.
- A Hold'em mode for `play.py`. Cheap to add later, not requested.

## 11. Open questions for you

1. Imperfect recall over buckets, as recommended in section 4? Or perfect?
2. Raise cap of 4 bets per street, or Leduc's 2 to keep the tree smaller?
3. Commit the generated `holdem_abstraction.json` (recommended, for
   reproducible CI), or build it on first use and gitignore it?
4. Keep the optional `--opponent call` baseline in `evaluate_vs_random.py`?
