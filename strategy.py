"""
Save, load, and read strategies.

    save_strategy(table, path)   -> write an average-strategy table as JSON
    load_strategy(path)          -> read it back
    leduc_summary(table)         -> human-readable chart of a Leduc strategy
    print_leduc_summary(table)   -> print it

A strategy table is the output of MCCFRTrainer.average_strategy_table():
    {info_set_key: {action: probability}}

The Leduc summary groups information sets by private card, then by street
(preflop, or flop with each possible public card), then by betting
situation, and shows the action frequencies with poker names (check/bet
when no bet is outstanding, call/raise/fold when facing a bet), so it reads
like a poker chart.

Command line:
    python3 strategy.py leduc_strategy.json
"""
import json
import sys

CARD_NAMES = {0: "J", 1: "Q", 2: "K"}

# Betting situations by round history, in the order they appear in play.
# (history, acting player, description)
_SITUATIONS = [
    ("", 0, "first to act"),
    ("c", 1, "facing a check"),
    ("r", 1, "facing a bet"),
    ("cr", 0, "checked, facing a bet"),
    ("rr", 0, "bet, facing a raise"),
    ("crr", 1, "bet, facing a raise"),
]

# Preflop lines that reach the flop, in the order they appear in play.
_PREFLOP_LINES = [
    ("cc", "check-check"),
    ("rc", "bet-call"),
    ("crc", "check, bet-call"),
    ("rrc", "bet, raise-call"),
    ("crrc", "check, bet, raise-call"),
]

_COLUMNS = ["check", "bet", "call", "raise", "fold"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_strategy(table, path):
    """Write a strategy table to `path` as JSON (sorted keys, indented)."""
    with open(path, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)


def load_strategy(path):
    """Read a strategy table written by save_strategy."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Leduc summary
# ---------------------------------------------------------------------------

def parse_leduc_key(key):
    """'card|pub|h0/h1' -> (card:int, pub:int|None, h0:str, h1:str)"""
    card, pub, hists = key.split("|")
    h0, h1 = hists.split("/")
    return int(card), (None if pub == "-" else int(pub)), h0, h1


def action_names(history):
    """Map the game's 'c'/'r'/'f' actions to poker names for a situation."""
    if history in ("", "c"):
        return {"c": "check", "r": "bet"}
    return {"c": "call", "r": "raise", "f": "fold"}


def _named_freqs(history, probs):
    """{action: prob} -> {poker name: prob} for the given situation."""
    names = action_names(history)
    return {names[a]: p for a, p in probs.items() if a in names}


def _fmt_row(label, freqs, label_width):
    cells = []
    for col in _COLUMNS:
        if col in freqs:
            cells.append(f"{100 * freqs[col]:5.1f}%")
        else:
            cells.append("     -")
    return f"  {label:<{label_width}} " + "  ".join(cells)


def _header(label_width):
    return f"  {'situation':<{label_width}} " + "  ".join(f"{c:>6}" for c in _COLUMNS)


def leduc_summary(table):
    """Return the human-readable chart for a Leduc strategy table as a string.

    Layout, per private card (K, Q, J):
        Preflop
          <situation rows>
        Flop, board J / Q / K
          after <preflop line>:
            <situation rows>
    Each row shows the frequency of every action available in that spot.
    """
    lines = []
    label_width = 30

    for card in sorted(CARD_NAMES, reverse=True):
        cname = CARD_NAMES[card]
        lines.append("=" * 76)
        lines.append(f"Private card: {cname}")
        lines.append("=" * 76)

        # Preflop
        lines.append("")
        lines.append("Preflop")
        lines.append(_header(label_width))
        for h, player, desc in _SITUATIONS:
            probs = table.get(f"{card}|-|{h}/")
            if probs is None:
                continue
            lines.append(_fmt_row(f"P{player} {desc}", _named_freqs(h, probs), label_width))

        # Flop, one block per public card, one sub-block per preflop line
        for pub in sorted(CARD_NAMES):
            pname = CARD_NAMES[pub]
            note = " (pair)" if pub == card else ""
            block = []
            for h0, line_desc in _PREFLOP_LINES:
                rows = []
                for h, player, desc in _SITUATIONS:
                    probs = table.get(f"{card}|{pub}|{h0}/{h}")
                    if probs is None:
                        continue
                    rows.append(_fmt_row(f"  P{player} {desc}", _named_freqs(h, probs), label_width))
                if rows:
                    block.append(f"  after {line_desc}:")
                    block.extend(rows)
            if block:
                lines.append("")
                lines.append(f"Flop, board {pname}{note}")
                lines.append(_header(label_width))
                lines.extend(block)
        lines.append("")

    return "\n".join(lines)


def print_leduc_summary(table, file=None):
    print(leduc_summary(table), file=file or sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    print_leduc_summary(load_strategy(sys.argv[1]))
