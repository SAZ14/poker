"""
Play heads-up Leduc Hold'em against a saved strategy.

    python3 train.py leduc --iterations 500000 --plus --save leduc.json
    python3 play.py --strategy leduc.json

Each hand shows your card, the board card when it is dealt, the pot, the
legal actions, and what the bot does. The bot samples its action from the
saved average strategy at its information set (uniform at random if the
info set is missing from the file). Seats alternate every hand so you are
player 0 (first to act) on odd hands and player 1 on even hands. Chips won
and lost accumulate across hands; type q at any prompt to quit.

Actions: c = check/call, r = bet/raise, f = fold (full words work too).
"""
import argparse
import random
import sys

import leduc
from strategy import load_strategy, action_names, CARD_NAMES

_INPUT_ALIASES = {
    "c": "c", "check": "c", "call": "c",
    "r": "r", "bet": "r", "raise": "r",
    "f": "f", "fold": "f",
    "q": "q", "quit": "q", "exit": "q",
}


class Quit(Exception):
    """Raised when the human asks to stop."""


def card_name(card):
    return CARD_NAMES[card]


def describe(action, history):
    """Poker name of a raw action in the given round history."""
    return action_names(history)[action]


def bot_action(table, state, rng):
    """Sample an action for the bot from the strategy table at this info set."""
    actions = state.legal_actions()
    probs = table.get(state.info_set_key())
    if probs is None:
        return rng.choice(actions)
    weights = [probs.get(a, 0.0) for a in actions]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(actions)
    r = rng.random() * total
    acc = 0.0
    for a, w in zip(actions, weights):
        acc += w
        if r < acc:
            return a
    return actions[-1]


def human_action(state, ask, out):
    """Prompt until the human gives a legal action. Raises Quit on 'q'."""
    actions = state.legal_actions()
    history = state.round_hists[state.round_idx]
    names = action_names(history)
    menu = ", ".join(f"[{a}] {names[a]}" for a in actions)
    while True:
        raw = ask(f"  Your move ({menu}, [q] quit): ")
        if raw is None:
            raise Quit()
        choice = _INPUT_ALIASES.get(raw.strip().lower())
        if choice == "q":
            raise Quit()
        if choice in actions:
            return choice
        print(f"  Not a legal action here. Choose one of: {menu}", file=out)


def play_hand(table, human_seat, rng, ask, out):
    """
    Play one hand. human_seat is 0 or 1; the bot takes the other seat.
    Returns the human's net chips for the hand. Raises Quit if the human
    quits mid-hand (no chips change hands).
    """
    bot_seat = 1 - human_seat
    state = leduc.LeducState.sample_root(rng)
    my_card = state.cards[human_seat]

    print(f"  You are player {human_seat} ({'first' if human_seat == 0 else 'second'} to act). "
          f"Your card: {card_name(my_card)}. Both ante {leduc.ANTE}.", file=out)

    while not state.is_terminal():
        player = state.current_player()

        if player == "CHANCE":
            state = state.deal_public_sample(rng)
            pair = " (you pair the board!)" if state.public == my_card else ""
            print(f"  --- Flop: board card is {card_name(state.public)}{pair}. "
                  f"Bet size is now {leduc.BET_SIZE[1]}.", file=out)
            continue

        history = state.round_hists[state.round_idx]
        pot = state.contrib[0] + state.contrib[1]

        if player == human_seat:
            to_call = state.contrib[bot_seat] - state.contrib[human_seat]
            facing = f", {to_call} to call" if to_call > 0 else ""
            print(f"  Pot: {pot} (you {state.contrib[human_seat]}, bot {state.contrib[bot_seat]}){facing}",
                  file=out)
            action = human_action(state, ask, out)
            print(f"  You {describe(action, history)}.", file=out)
        else:
            action = bot_action(table, state, rng)
            print(f"  Bot {describe(action, history)}s.", file=out)

        state = state.next_state(action)

    result = state.utility(human_seat)
    if state.folded is not None:
        who = "You" if state.folded == human_seat else "Bot"
        print(f"  {who} folded.", file=out)
    else:
        bot_card = state.cards[bot_seat]
        board = card_name(state.public) if state.public is not None else "-"
        print(f"  Showdown: you {card_name(my_card)}, bot {card_name(bot_card)}, board {board}.", file=out)

    if result > 0:
        print(f"  You win {_chips(result)}.", file=out)
    elif result < 0:
        print(f"  You lose {_chips(-result)}.", file=out)
    else:
        print("  Split pot.", file=out)
    return result


def _chips(n):
    return f"{n:g} chip{'' if n == 1 else 's'}"


def play_session(table, rng, ask, out, max_hands=None):
    """Play hands until the human quits or max_hands is reached.
    Returns (hands_played, cumulative_chips)."""
    hands = 0
    total = 0.0
    try:
        while max_hands is None or hands < max_hands:
            human_seat = hands % 2
            print(f"\n=== Hand {hands + 1} ===", file=out)
            total += play_hand(table, human_seat, rng, ask, out)
            hands += 1
            print(f"  Running total: {total:+g} chips over {hands} hand{'s' if hands != 1 else ''}.",
                  file=out)
    except Quit:
        print("\nQuitting.", file=out)
    return hands, total


def _stdin_ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play heads-up Leduc Hold'em against a saved strategy.")
    parser.add_argument("--strategy", required=True, metavar="PATH",
                        help="JSON strategy file written by train.py --save")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for deals and bot sampling")
    parser.add_argument("--hands", type=int, default=None, help="stop after this many hands")
    args = parser.parse_args(argv)

    table = load_strategy(args.strategy)
    rng = random.Random(args.seed)
    print(f"Loaded strategy with {len(table)} info sets from {args.strategy}.")
    print("Actions: c = check/call, r = bet/raise, f = fold, q = quit.")

    hands, total = play_session(table, rng, _stdin_ask, sys.stdout, max_hands=args.hands)
    if hands:
        print(f"\nFinal: {total:+g} chips over {hands} hand{'s' if hands != 1 else ''} "
              f"({total / hands:+.3f} per hand).")


if __name__ == "__main__":
    main()
