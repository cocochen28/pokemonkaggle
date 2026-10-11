"""Monte Carlo lookahead on top of the rule-based policy.

For a decision with several single-pick options, every option is tried in the engine's search
simulator: hidden cards (my deck/prizes, the opponent's deck/hand/prizes) are filled in with a
plausible guess, the option is played, and the game is finished by the rule-based policy for both
players. The option that wins most often is picked.
"""
import ctypes
import itertools
import json
import math
import random
import time
from collections import Counter

from cg.sim import lib

lib.AgentStart.restype = ctypes.c_void_p
_agent_ptr = None

# Card types / areas (raw ints, same as main.py)
POKEMON, BASIC_ENERGY, SPECIAL_ENERGY = 0, 5, 6
C_MAIN = 0


def _arr(xs):
    return (ctypes.c_int * len(xs))(*xs)


def _begin(obs, my_deck, my_prize, o_deck, o_prize, o_hand, o_active):
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = lib.AgentStart()
    sbi = obs["search_begin_input"]
    r = json.loads(lib.SearchBegin(_agent_ptr, sbi.encode("ascii"), len(sbi), _arr(my_deck), _arr(my_prize),
                                   _arr(o_deck), _arr(o_prize), _arr(o_hand), _arr(o_active), 0))
    if r["error"]:
        raise RuntimeError(f"SearchBegin error {r['error']}")
    return r["state"]


def _step(sid, sel):
    r = json.loads(lib.SearchStep(_agent_ptr, sid, _arr(sel), len(sel)))
    if r["error"]:
        raise RuntimeError(f"SearchStep error {r['error']}")
    return r["state"]


def _end():
    if _agent_ptr is not None:
        lib.SearchEnd(_agent_ptr)


# ---------------------------------------------------------------------------
# Opponent modelling: remember every opponent card we have seen this match (same decks all Bo3).
# ---------------------------------------------------------------------------
class OppModel:
    def __init__(self):
        self.max_seen = Counter()  # card id -> most copies seen at once

    def observe(self, obs, opp):
        visible = Counter(_visible_ids(obs["current"]["players"][opp], obs["current"], opp))
        for l in obs.get("logs") or []:
            if l.get("playerIndex") == opp and l.get("cardId"):
                visible.setdefault(l["cardId"], 1)
        for cid, n in visible.items():
            if n > self.max_seen[cid]:
                self.max_seen[cid] = n


def _visible_ids(p, cur, owner):
    ids = [c["id"] for c in p["discard"]]
    for pk in [x for x in p["active"] if x] + list(p["bench"]):
        ids.append(pk["id"])
        ids += [c["id"] for c in pk["energyCards"]] + [c["id"] for c in pk["tools"]]
        ids += [c["id"] for c in pk["preEvolution"]]
    ids += [c["id"] for c in p["prize"] if c]
    if p.get("hand"):
        ids += [c["id"] for c in p["hand"]]
    for c in cur.get("stadium") or []:
        if c and c.get("playerIndex") == owner:
            ids.append(c["id"])
    return ids


class Searcher:
    def __init__(self, main_mod):
        self.m = main_mod
        self.opp_model = OppModel()
        self.used = 0.0  # seconds spent searching this process (fallback when no overage info)

    # -- hidden information --------------------------------------------------
    def _guess_opp_deck(self, opp_player_cards):
        """A 60-card guess of the opponent's deck from what we've seen."""
        CARDS = self.m.CARDS
        seen = self.opp_model.max_seen
        deck = []
        for cid, n in seen.items():
            deck += [cid] * n
        # Scale up: non-energy cards toward 3 copies, then fill with their most common basic energy.
        for cid in list(seen):
            c = CARDS.get(cid)
            if c and c.cardType != BASIC_ENERGY and not c.aceSpec:
                deck += [cid] * max(0, 3 - seen[cid])
        energies = [cid for cid in seen if CARDS[cid].cardType == BASIC_ENERGY]
        if not energies:
            types = [CARDS[cid].energyType for cid in seen if CARDS[cid].cardType == POKEMON]
            energies = [t for t in types if 1 <= t <= 9] or [3]  # basic energy id == energy type
        while len(deck) < 60:
            deck.append(random.choice(energies))
        return deck

    def _fill(self, pool, n, fallback):
        out = []
        for _ in range(n):
            out.append(pool.pop() if pool else fallback)
        return out

    def determinize(self, obs):
        cur = obs["current"]
        me = cur["yourIndex"]
        opp = 1 - me
        mp, op = cur["players"][me], cur["players"][opp]
        CARDS = self.m.CARDS

        # Mine: deck list minus everything visible.
        pool = list(self.m.MY_DECK)
        for cid in _visible_ids(mp, cur, me):
            if cid in pool:
                pool.remove(cid)
        random.shuffle(pool)
        my_fallback = next((c for c in self.m.MY_DECK if CARDS[c].cardType == BASIC_ENERGY), self.m.MY_DECK[0])
        my_prize = [c["id"] if c else (pool.pop() if pool else my_fallback) for c in mp["prize"]]
        my_deck = self._fill(pool, mp["deckCount"], my_fallback)

        # Opponent: guessed deck minus everything visible.
        guess = self._guess_opp_deck(None)
        for cid in _visible_ids(op, cur, opp):
            if cid in guess:
                guess.remove(cid)
        random.shuffle(guess)
        o_fallback = next((c for c in guess if CARDS[c].cardType == BASIC_ENERGY), 3)
        o_active = []
        if op["active"] and op["active"][0] is None:
            basics = [c for c in guess if CARDS[c].cardType == POKEMON and CARDS[c].basic]
            b = basics[0] if basics else next(c.cardId for c in CARDS.values() if c.cardType == POKEMON and c.basic)
            if b in guess:
                guess.remove(b)
            o_active = [b]
        o_hand = self._fill(guess, op["handCount"], o_fallback)
        o_prize = [c["id"] if c else (guess.pop() if guess else o_fallback) for c in op["prize"]]
        o_deck = self._fill(guess, op["deckCount"], o_fallback)
        return my_deck, my_prize, o_deck, o_prize, o_hand, o_active

    # -- rollouts ----------------------------------------------------------------
    def rollout(self, state, me, max_steps=600):
        """Finish the game with the rule-based policy. Returns my result in [0, 1]."""
        policy = self.m.policy
        so = state["observation"]
        sid = state["searchId"]
        for _ in range(max_steps):
            cur = so["current"]
            if cur["result"] != -1:
                return 1.0 if cur["result"] == me else 0.5 if cur["result"] == 2 else 0.0
            st = _step(sid, policy(so))
            sid, so = st["searchId"], st["observation"]
        return self.static_eval(so["current"], me)

    @staticmethod
    def static_eval(cur, me):
        mine = len(cur["players"][me]["prize"])
        theirs = len(cur["players"][1 - me]["prize"])
        return 0.5 + 0.08 * (theirs - mine)

    @staticmethod
    def candidates(sel, default):
        """Selections worth comparing. `default` (the rule-based pick) is always candidate 0."""
        n = len(sel["option"])
        lo, hi = sel["minCount"], sel["maxCount"]
        default = list(default)
        if lo == hi == 1:
            return [default] + [[i] for i in range(n) if i != default[0]]
        k = len(default)
        if k == 0 or n > 10:
            return [default]
        # Same number of picks as the rule-based choice; all combinations if few, else single swaps.
        if math.comb(n, k) <= 10:
            combos = [sorted(c) for c in itertools.combinations(range(n), k)]
            return [default] + [c for c in combos if c != sorted(default)]
        out = [default]
        for j in range(k):
            for alt in range(n):
                if alt not in default:
                    out.append(default[:j] + [alt] + default[j + 1:])
        return out[:10]

    def choose(self, obs, budget, default):
        """Return the selection (list of option indexes) that wins most often in simulation."""
        me = obs["current"]["yourIndex"]
        cands = self.candidates(obs["select"], default)
        if len(cands) < 2:
            return default
        m = len(cands)
        deadline = time.time() + budget
        wins = [0.0] * m
        visits = [0] * m
        alive = list(range(m))
        t0 = time.time()
        try:
            rnd = 0
            while time.time() < deadline or rnd == 0:
                for i in list(alive):
                    hidden = self.determinize(obs)
                    root = _begin(obs, *hidden)
                    try:
                        st = _step(root["searchId"], cands[i])
                    except RuntimeError:  # not a legal selection
                        _end()
                        if i == 0:
                            return default
                        alive.remove(i)
                        continue
                    wins[i] += self.rollout(st, me)
                    visits[i] += 1
                    _end()
                rnd += 1
                if len(alive) < 2:
                    return default
                # Successive halving: spend the remaining time on the most promising candidates.
                if rnd in (3, 6, 12, 24) and len(alive) > 2:
                    alive.sort(key=lambda i: -wins[i] / visits[i])
                    keep = alive[:max(2, (len(alive) + 1) // 2)]
                    if 0 not in keep:
                        keep.append(0)
                    alive = keep
        except Exception:
            _end()
            if self.m.DEBUG:
                raise
            return None
        finally:
            self.used += time.time() - t0
        rates = [wins[i] / visits[i] if visits[i] else 0 for i in range(m)]
        best = max(alive, key=lambda i: rates[i])
        # Only overrule the rule-based pick when the evidence is clear (rollouts are noisy).
        if rates[best] - rates[0] < self.margin(min(visits[best], visits[0])):
            return default
        return cands[best]

    @staticmethod
    def margin(visits):
        return 0.5 / (visits ** 0.5)
