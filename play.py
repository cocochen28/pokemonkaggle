"""Local test harness: play games between two agents and report win rates.

Usage:
    python play.py                          # 100 games: main.agent vs random, both using deck.csv
    python play.py 500                      # 500 games
    python play.py 1 --verbose              # one game, print every decision
    python play.py 200 decks/a.csv decks/b.csv   # bot with deck A vs bot with deck B
    python play.py 200 decks/a.csv decks/a.csv --b=bots/v1.py   # current bot vs an older bot
"""
import importlib.util
import os
import random
import sys
from collections import Counter

from cg.game import battle_start, battle_select, battle_finish
from cg.api import SelectContext, OptionType
import main


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        deck = [int(x) for x in f.read().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} has {len(deck)} cards, needs 60")
    return deck


def random_agent(obs_dict: dict) -> list[int]:
    n = len(obs_dict["select"]["option"])
    return random.sample(range(n), obs_dict["select"]["maxCount"])


def load_bot(path: str):
    """Load an agent module from a .py file (e.g. an older saved version in bots/)."""
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bot_with_deck(deck, module=main):
    """module.agent, but told which deck it's holding (so two bots with different decks can play)."""
    def agent(obs_dict):
        if hasattr(module, "set_deck"):
            module.set_deck(deck)
        return module.agent(obs_dict)
    return agent


def play_one(agents, decks, verbose=False, max_steps=5000):
    """Play one game. agents[i] plays player index i. Returns winner index (0/1, 2=draw, -1=aborted)."""
    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"Deck rejected: player {start.errorPlayer}, error {start.errorType}")
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur is not None and cur["result"] != -1:
                return cur["result"]
            who = cur["yourIndex"]
            choice = agents[who](obs)
            if verbose:
                sel = obs["select"]
                opts = [OptionType(o["type"]).name for o in sel["option"]]
                print(f"T{cur['turn']} P{who} {SelectContext(sel['context']).name}: {opts} -> {choice}")
            obs = battle_select(choice)
        return -1
    finally:
        battle_finish()


def match(n, agent_a, deck_a, agent_b, deck_b, verbose=False):
    """Play n games, alternating seats. Returns Counter with keys 'A', 'B', 'draw'."""
    results = Counter()
    for g in range(n):
        if g % 2 == 0:
            w = play_one([agent_a, agent_b], [deck_a, deck_b], verbose)
            results["A" if w == 0 else "B" if w == 1 else "draw"] += 1
        else:
            w = play_one([agent_b, agent_a], [deck_b, deck_a], verbose)
            results["A" if w == 1 else "B" if w == 0 else "draw"] += 1
    return results


def main_cli():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose = "--verbose" in sys.argv
    n = int(args[0]) if args and args[0].isdigit() else 100
    paths = [a for a in args if a.endswith(".csv")]

    bot_paths = {a[2]: a[4:] for a in sys.argv[1:] if a.startswith(("--a=", "--b="))}
    mod_a = load_bot(bot_paths["a"]) if "a" in bot_paths else main
    mod_b = load_bot(bot_paths["b"]) if "b" in bot_paths else main

    if len(paths) == 2:
        deck_a, deck_b = load_deck(paths[0]), load_deck(paths[1])
        r = match(n, bot_with_deck(deck_a, mod_a), deck_a, bot_with_deck(deck_b, mod_b), deck_b, verbose)
        print(f"{n} games: {paths[0]} won {r['A']}, {paths[1]} won {r['B']}, draws {r['draw']}"
              f"  -> {paths[0]} win rate = {r['A'] / n:.1%}")
    else:
        deck = load_deck(paths[0]) if paths else main.read_deck_csv()
        r = match(n, bot_with_deck(deck), deck, random_agent, deck, verbose)
        print(f"{n} games vs random: bot won {r['A']}, random won {r['B']}, draws {r['draw']}"
              f"  -> bot win rate = {r['A'] / n:.1%}")


if __name__ == "__main__":
    main_cli()
