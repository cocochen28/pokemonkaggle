import os
import random
import re
import sys

from cg.api import all_card_data, all_attack

# ---------------------------------------------------------------------------
# Constants (mirror the IntEnums in cg/api.py; plain ints because we work on the raw dict)
# ---------------------------------------------------------------------------
# AreaType
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM, LOOKING = 1, 2, 3, 4, 5, 6, 7, 12
# CardType
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM_CARD, BASIC_ENERGY, SPECIAL_ENERGY = range(7)
# EnergyType
COLORLESS, RAINBOW, TEAM_ROCKET = 0, 10, 11
TYPE_LETTER = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8, "N": 9}
# OptionType
O_NUMBER, O_YES, O_NO, O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY = 0, 1, 2, 3, 4, 5, 6
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK, O_END = 7, 8, 9, 10, 11, 12, 13, 14
O_SKILL, O_SPECIAL_CONDITION = 15, 16
# SelectContext (only the ones we treat specially)
C_MAIN, C_SETUP_ACTIVE, C_SETUP_BENCH, C_SWITCH, C_TO_ACTIVE = 0, 1, 2, 3, 4
C_TO_BENCH, C_TO_FIELD, C_TO_HAND, C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE = 5, 6, 7, 8, 9, 10, 11
C_DAMAGE_COUNTER, C_DAMAGE_COUNTER_ANY, C_DAMAGE = 13, 14, 15
C_REMOVE_DAMAGE_COUNTER, C_HEAL = 16, 17
C_ATTACH_FROM, C_ATTACH_TO = 21, 22
C_DISCARD_ENERGY_CARD, C_DISCARD_TOOL_CARD = 26, 27
C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY, C_TO_HAND_ENERGY, C_TO_DECK_ENERGY = 29, 30, 31, 32
C_ATTACK, C_DISABLE_ATTACK = 35, 36
C_REMOVE_DAMAGE_COUNTER_COUNT = 40
C_IS_FIRST = 41
C_RECOVER_SPECIAL_CONDITION = 48

# Contexts where picking a card is bad for its owner -> pick as few as allowed, lowest value first.
BAD_FOR_OWNER = {C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE, C_DISCARD_ENERGY_CARD,
                 C_DISCARD_TOOL_CARD, C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY,
                 C_TO_HAND_ENERGY, C_TO_DECK_ENERGY, C_DAMAGE_COUNTER, C_DAMAGE_COUNTER_ANY, C_DAMAGE}

# ---------------------------------------------------------------------------
# Card database
# ---------------------------------------------------------------------------
CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


MY_DECK = []
# Names of Pokémon that something in my deck evolves from -> list of evolution card IDs.
EVOLUTIONS = {}


def set_deck(deck: list[int]) -> None:
    """Tell the agent which deck it is playing (used to know evolutions and what's left in the deck)."""
    global MY_DECK, EVOLUTIONS
    MY_DECK = list(deck)
    EVOLUTIONS = {}
    for cid in set(MY_DECK):
        c = CARDS[cid]
        if c.cardType == POKEMON and c.evolvesFrom:
            EVOLUTIONS.setdefault(c.evolvesFrom, []).append(cid)


set_deck(read_deck_csv())

# Every evolution in the card pool (used when the policy plays the opponent inside search rollouts).
EVOLUTIONS_ALL = {}
for _c in CARDS.values():
    if _c.cardType == POKEMON and _c.evolvesFrom:
        EVOLUTIONS_ALL.setdefault(_c.evolvesFrom, []).append(_c.cardId)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def missing_energy(cost: list[int], energies: list[int]) -> int:
    """How many more Energy are needed to pay `cost` with the attached `energies`."""
    pool = list(energies)
    missing = 0
    colorless = 0
    for t in cost:
        if t == COLORLESS:
            colorless += 1
            continue
        if t in pool:
            pool.remove(t)
        elif RAINBOW in pool:
            pool.remove(RAINBOW)
        elif t in (5, 7) and TEAM_ROCKET in pool:
            pool.remove(TEAM_ROCKET)
        else:
            missing += 1
    return missing + max(0, colorless - len(pool))


def prize_value(cid: int) -> int:
    c = CARDS.get(cid)
    if c is None:
        return 1
    if c.megaEx:
        return 3
    if c.ex:
        return 2
    return 1


class Ctx:
    """Everything about the current decision, with lookup helpers."""

    def __init__(self, obs: dict):
        self.obs = obs
        self.sel = obs["select"]
        self.cur = obs["current"]
        self.me = self.cur["yourIndex"]
        self.opp = 1 - self.me
        self.my = self.cur["players"][self.me]
        self.their = self.cur["players"][self.opp]
        self.hand = self.my["hand"] or []
        self.evolutions = EVOLUTIONS if REAL_ME is None or self.me == REAL_ME else EVOLUTIONS_ALL

    # -- locating cards -----------------------------------------------------
    def card_at(self, area, index, player=None):
        """Return the card/pokemon dict at a location (or None)."""
        p = self.cur["players"][self.me if player is None else player]
        try:
            if area == HAND:
                return p["hand"][index]
            if area == DECK:
                return (self.sel.get("deck") or [])[index]
            if area == DISCARD:
                return p["discard"][index]
            if area == ACTIVE:
                return p["active"][index]
            if area == BENCH:
                return p["bench"][index]
            if area == PRIZE:
                return p["prize"][index]
            if area == LOOKING:
                return (self.cur.get("looking") or [])[index]
            if area == STADIUM:
                return self.cur["stadium"][index]
        except (IndexError, TypeError, KeyError):
            return None
        return None

    def in_play(self, player=None):
        p = self.cur["players"][self.me if player is None else player]
        return [x for x in p["active"] if x] + list(p["bench"])

    def my_active(self):
        a = self.my["active"]
        return a[0] if a else None

    def their_active(self):
        a = self.their["active"]
        return a[0] if a else None

    def seen_cards(self):
        """All of my card IDs that are visible (not in deck/prizes)."""
        ids = [c["id"] for c in self.hand] + [c["id"] for c in self.my["discard"]]
        for pk in self.in_play():
            ids.append(pk["id"])
            ids += [c["id"] for c in pk["energyCards"]] + [c["id"] for c in pk["tools"]]
            ids += [c["id"] for c in pk["preEvolution"]]
        return ids

    def unseen_deck_cards(self):
        rest = list(MY_DECK)
        for cid in self.seen_cards():
            if cid in rest:
                rest.remove(cid)
        return rest

    # -- attack evaluation --------------------------------------------------
    def attack_damage(self, attack_id, attacker, defender) -> float:
        """Rough expected damage of an attack against a defender (Pokémon dicts)."""
        atk = ATTACKS.get(attack_id)
        if atk is None:
            return 0
        dmg = atk.damage
        text = atk.text or ""
        if dmg == 0 and "damage" in text:
            dmg = self._variable_damage(text, attacker)
        if defender is not None and dmg > 0:
            a_type = CARDS[attacker["id"]].energyType
            d = CARDS.get(defender["id"])
            if d is not None:
                if d.weakness == a_type:
                    dmg *= 2
                elif d.resistance == a_type:
                    dmg = max(0, dmg - 30)
        return dmg

    def _variable_damage(self, text, attacker) -> float:
        # "Discard the top 6 cards of your deck, and this attack does 100 damage for each Basic {W} Energy card..."
        m = re.search(r"top (\d+) cards of your deck.*?(\d+) damage for each Basic \{(\w)\} Energy", text, re.S)
        if m:
            n, per, t = int(m.group(1)), int(m.group(2)), TYPE_LETTER.get(m.group(3), -1)
            rest = self.unseen_deck_cards()
            if not rest:
                return 0
            frac = sum(1 for c in rest if CARDS[c].cardType == BASIC_ENERGY and CARDS[c].energyType == t) / len(rest)
            return per * min(n, self.my["deckCount"]) * frac
        # "does 20 damage for each Basic {W} Energy card in your discard pile"
        m = re.search(r"(\d+) damage for each Basic \{(\w)\} Energy card in your discard pile", text)
        if m:
            per, t = int(m.group(1)), TYPE_LETTER.get(m.group(2), -1)
            return per * sum(1 for c in self.my["discard"]
                             if CARDS[c["id"]].cardType == BASIC_ENERGY and CARDS[c["id"]].energyType == t)
        # "does N damage for each Energy attached to this Pokémon"
        m = re.search(r"(\d+) damage for each Energy attached to this", text)
        if m:
            return int(m.group(1)) * len(attacker["energies"])
        m = re.search(r"(\d+) damage for each", text)
        if m:
            return int(m.group(1)) * 2  # unknown count: guess 2
        return 0

    def best_ready_damage(self, pk, defender=None) -> float:
        """Best damage this Pokémon can do right now with its attached energy (0 if it can't attack)."""
        if pk is None:
            return 0
        defender = defender or self.their_active()
        best = 0
        for aid in CARDS[pk["id"]].attacks:
            atk = ATTACKS.get(aid)
            if atk and missing_energy(atk.energies, pk["energies"]) == 0:
                best = max(best, self.attack_damage(aid, pk, defender))
        return best

    def potential_attacks(self, pk):
        """Attacks this Pokémon has or will have after evolving (evolutions found in my deck)."""
        ids = [pk["id"]] + self.evolutions.get(CARDS[pk["id"]].name, [])
        for e in list(ids[1:]):
            ids += self.evolutions.get(CARDS[e].name, [])
        out = []
        for cid in ids:
            for aid in CARDS[cid].attacks:
                atk = ATTACKS.get(aid)
                if atk:
                    out.append(atk)
        return out

    def energy_helpfulness(self, pk, energy_types) -> float:
        """How useful it is to give `energy_types` to Pokémon `pk` (0 = useless)."""
        best = 0
        for atk in self.potential_attacks(pk):
            before = missing_energy(atk.energies, pk["energies"])
            after = missing_energy(atk.energies, pk["energies"] + energy_types)
            if after < before:
                dmg = atk.damage or 80  # variable-damage attacks are usually strong
                score = 1 + min(dmg, 300) / 300 + (1 if after == 0 else 0)
                best = max(best, score)
        return best

    def can_ko_benched(self) -> bool:
        """Can my Active Pokémon knock out one of the opponent's Benched Pokémon right now?"""
        me = self.my_active()
        return any(self.best_ready_damage(me, pk) >= pk["hp"] for pk in self.their["bench"])

    def bench_much_better(self) -> bool:
        """Is one of my Benched Pokémon a much better Active Pokémon than the current one?"""
        bench = self.my["bench"]
        if not bench:
            return False
        return max(self.pokemon_strength(b) for b in bench) > self.pokemon_strength(self.my_active()) + 60

    def pokemon_strength(self, pk) -> float:
        """How good this Pokémon would be in the Active Spot right now."""
        if pk is None:
            return -100
        ready = self.best_ready_damage(pk)
        s = ready + pk["hp"] * 0.5
        if ready > 0:
            s += 100
        their = self.their_active()
        if their and ready >= their["hp"]:
            s += 80
        return s

    # -- generic card value (for search / discard / to-hand) ----------------
    def card_value(self, cid, from_hand=False) -> float:
        c = CARDS.get(cid)
        if c is None:
            return 0
        in_play_names = {CARDS[p["id"]].name for p in self.in_play()}
        hand_names = {CARDS[h["id"]].name for h in self.hand}
        n_in_play = len(self.in_play())
        energy_in_hand = sum(1 for h in self.hand if CARDS[h["id"]].cardType in (BASIC_ENERGY, SPECIAL_ENERGY))
        v = 0
        if c.cardType == POKEMON:
            if c.basic:
                v = 90 if n_in_play <= 2 else 70 if n_in_play < 4 else 30
                if c.name in self.evolutions:
                    v += 10
            elif c.evolvesFrom in in_play_names:
                bases = sum(1 for p in self.in_play() if CARDS[p["id"]].name == c.evolvesFrom)
                v = 95 if bases > self._copies_in_hand(cid, from_hand) else 40
            elif c.evolvesFrom in hand_names:
                v = 65
            else:
                v = 45
            if c.megaEx or c.ex:
                v += 5
        elif c.cardType == SUPPORTER:
            v = 60
        elif c.cardType == ITEM:
            v = 50
        elif c.cardType == TOOL:
            v = 40
        elif c.cardType == STADIUM_CARD:
            v = 30
        else:  # energy
            v = 55 if energy_in_hand == 0 else 25
        if c.aceSpec:
            v += 10
        # Extra copies of something already in hand are worth less (except energy, handled above).
        if c.cardType not in (BASIC_ENERGY, SPECIAL_ENERGY):
            v -= 20 * self._copies_in_hand(cid, from_hand)
        return v

    def _copies_in_hand(self, cid, from_hand) -> int:
        """Copies of this card in hand, not counting the card itself if it's one of them."""
        return sum(1 for h in self.hand if h["id"] == cid) - (1 if from_hand else 0)


# ---------------------------------------------------------------------------
# Main-phase scoring
# ---------------------------------------------------------------------------
def score_main(ctx: Ctx, opt: dict) -> float:
    t = opt["type"]
    if t == O_END:
        return 0
    if t == O_EVOLVE:
        c = CARDS[ctx.card_at(opt["area"], opt["index"])["id"]]
        return 100 + (5 if c.megaEx or c.ex else 0)
    if t == O_PLAY:
        cid = ctx.hand[opt["index"]]["id"]
        if "turn ends" in skill_text(cid):
            return score_turn_ender(ctx)
        return score_play(ctx, cid)
    if t == O_ATTACH:
        card = ctx.card_at(opt["area"], opt["index"])
        target = ctx.card_at(opt["inPlayArea"], opt["inPlayIndex"])
        if card is None or target is None:
            return 30
        c = CARDS[card["id"]]
        if c.cardType == TOOL:
            return 45 + (5 if opt["inPlayArea"] == ACTIVE else 0)
        etypes = [c.energyType] if c.cardType == BASIC_ENERGY else [COLORLESS]
        return score_energy_attach(ctx, target, opt["inPlayArea"] == ACTIVE, etypes)
    if t == O_ABILITY:
        card = ctx.card_at(opt["area"], opt["index"])
        if card and "turn ends" in skill_text(card["id"]):
            return score_turn_ender(ctx)
        return 65
    if t == O_RETREAT:
        return score_retreat(ctx)
    if t == O_ATTACK:
        attacker = ctx.my_active()
        defender = ctx.their_active()
        dmg = ctx.attack_damage(opt["attackId"], attacker, defender)
        s = 10 + min(dmg, 400) / 20
        if defender and dmg >= defender["hp"]:
            s += 15 + 3 * prize_value(defender["id"])
        return s
    if t == O_DISCARD:
        return -5
    return 1


def skill_text(cid: int) -> str:
    c = CARDS.get(cid)
    return " ".join(s.text for s in c.skills) if c else ""


def score_turn_ender(ctx: Ctx) -> float:
    """Effects that end the turn (Lumiose City, Boxed Order...): last thing to do, never instead of attacking."""
    can_attack = any(o["type"] == O_ATTACK for o in ctx.sel["option"])
    return -1 if can_attack else 5


def score_energy_attach(ctx: Ctx, target: dict, is_active: bool, etypes: list[int]) -> float:
    """Focus energy: get the Active Pokémon attacking first, then power up the next attacker on the bench."""
    help_ = ctx.energy_helpfulness(target, etypes)
    if is_active:
        if ctx.best_ready_damage(target) == 0:
            if help_ > 0:
                return 100 + help_
            # Can't help it attack, but energy still pays for retreating to a better Pokémon.
            if CARDS[target["id"]].retreatCost > len(target["energies"]) and ctx.bench_much_better():
                return 90
            return 30
        return 70 + 10 * help_ if help_ else 30
    return 70 + 10 * help_ if help_ else 25


def score_play(ctx: Ctx, cid: int) -> float:
    c = CARDS[cid]
    hand_size = len(ctx.hand)
    text = " ".join(s.text for s in c.skills)
    if c.cardType == POKEMON:  # basic Pokémon onto the bench
        n = len(ctx.in_play())
        return 90 if n < 3 else 60 if n < 5 else 35

    # Situational cards: only worth playing when they actually do something.
    if "Switch your Active Pokémon with 1 of your Benched" in text:
        return 74 if ctx.bench_much_better() else -5
    if "Switch in 1 of your opponent" in text:
        return 66 if ctx.can_ko_benched() else -5
    if "During this turn, attacks used by your" in text:
        first_turn = ctx.cur["turn"] == 1  # the player going first can't attack on turn 1
        return 58 if not first_turn and ctx.best_ready_damage(ctx.my_active()) > 0 else -5
    if text.startswith("Heal"):
        a = ctx.my_active()
        return 55 if a and a["maxHp"] - a["hp"] >= 50 else -5

    if c.cardType == SUPPORTER:
        if "Shuffle your hand into your deck" in text or "Discard your hand" in text:
            return 60 if hand_size <= 4 else 40 if hand_size <= 6 else 5
        return 60
    if c.cardType == ITEM:
        return 75
    if c.cardType == TOOL:
        return 45
    if c.cardType == STADIUM_CARD:
        return 40
    return 30


def score_retreat(ctx: Ctx) -> float:
    return 72 if ctx.bench_much_better() else -10  # before attacking, after setting up


# ---------------------------------------------------------------------------
# Non-main selections
# ---------------------------------------------------------------------------
def score_option(ctx: Ctx, opt: dict, context: int) -> float:
    """Higher = more desirable to pick in this context (from my point of view)."""
    t = opt["type"]
    if t == O_YES:
        return 1
    if t == O_NO:
        return 0
    if t == O_NUMBER:
        return opt.get("number") or 0  # draw/place/remove as many as allowed
    if t == O_ATTACK:
        dmg = ctx.attack_damage(opt["attackId"], ctx.my_active(), ctx.their_active())
        return dmg
    if t == O_SPECIAL_CONDITION:
        # 0 poison,1 burn,2 sleep,3 paralyze,4 confuse; inflict paralysis/sleep, recover the same first
        return {3: 5, 2: 4, 4: 3, 1: 2, 0: 1}.get(opt.get("specialConditionType"), 0)
    if t == O_SKILL:
        return 0

    owner = opt.get("playerIndex", ctx.me)
    mine = owner == ctx.me
    area = opt.get("area")
    obj = ctx.card_at(area, opt.get("index", 0), owner) if area is not None else None
    if obj is None:
        return 0

    # Attached cards (energy / tool) – value of what they're attached to.
    if t in (O_ENERGY_CARD, O_ENERGY, O_TOOL_CARD):
        importance = 2 if area == ACTIVE else 1
        return importance if mine else -importance  # higher = more painful for opponent to lose

    cid = obj["id"]
    c = CARDS.get(cid)
    if c is None:
        return 0
    is_pokemon_in_play = area in (ACTIVE, BENCH)

    if context == C_SETUP_ACTIVE:
        # Prefer something that can attack soon and retreat cheaply, so we don't get stuck.
        cheapest = min((len(ATTACKS[a].energies) for a in c.attacks if a in ATTACKS), default=5)
        return c.hp / 10 - 15 * c.retreatCost - 10 * cheapest
    if context in (C_SWITCH, C_TO_ACTIVE):
        if mine:
            return ctx.pokemon_strength(obj) if is_pokemon_in_play else c.hp
        # Gusting an opponent's Pokémon: prefer one we can KO, then high prize value, then low HP.
        dmg = ctx.best_ready_damage(ctx.my_active(), obj) if is_pokemon_in_play else 0
        ko = is_pokemon_in_play and dmg >= obj["hp"]
        return (1000 if ko else 0) + 100 * prize_value(cid) - (obj.get("hp", c.hp) if is_pokemon_in_play else 0)
    if context in (C_DAMAGE_COUNTER, C_DAMAGE_COUNTER_ANY, C_DAMAGE):
        if mine:
            return -1000
        hp = obj.get("hp", 999) if is_pokemon_in_play else 999
        return 500 - hp + 100 * prize_value(cid) + (50 if area == ACTIVE else 0)
    if context in (C_HEAL, C_REMOVE_DAMAGE_COUNTER):
        if not mine:
            return -1000
        return (obj["maxHp"] - obj["hp"]) if is_pokemon_in_play else 0
    if context == C_ATTACH_FROM:
        # Choosing a Pokémon to attach something to.
        if not mine:
            return -100
        if not is_pokemon_in_play:
            return 0
        card = ctx.sel.get("contextCard")
        if card:
            cc = CARDS[card["id"]]
            if cc.cardType in (BASIC_ENERGY, SPECIAL_ENERGY):
                et = [cc.energyType] if cc.cardType == BASIC_ENERGY else [COLORLESS]
                return ctx.energy_helpfulness(obj, et) * 10 + (3 if area == ACTIVE else 0)
        return (5 if area == ACTIVE else 0) + ctx.best_ready_damage(obj) / 100
    if context == C_ATTACH_TO:
        return ctx.card_value(cid) if mine else 0
    if is_pokemon_in_play:
        # Generic in-play target: for my Pokémon, stronger is better; for theirs, hurting the best one is better.
        return ctx.pokemon_strength(obj) if mine else 100 * prize_value(cid) + (50 if area == ACTIVE else 0)
    v = ctx.card_value(cid, from_hand=(area == HAND and mine))
    return v if mine else -v


def choose(ctx: Ctx) -> list[int]:
    sel = ctx.sel
    opts = sel["option"]
    lo, hi = sel["minCount"], sel["maxCount"]
    context = sel["context"]
    n = len(opts)

    if context == C_MAIN:
        scores = [score_main(ctx, o) for o in opts]
        return [max(range(n), key=lambda i: scores[i])][: hi] or list(range(lo))

    if context == C_IS_FIRST:
        return [next(i for i, o in enumerate(opts) if o["type"] == O_YES)]

    # Skill ordering: any order works, keep given order.
    if opts and opts[0]["type"] == O_SKILL:
        return list(range(hi))

    scores = [score_option(ctx, o, context) for o in opts]

    # Energy payment (e.g. retreat cost): cover remainEnergyCost with the least important energies.
    if opts and opts[0]["type"] == O_ENERGY and sel.get("remainEnergyCost", 0) > 0 and context not in BAD_FOR_OWNER:
        order = sorted(range(n), key=lambda i: scores[i])
        picked, paid = [], 0
        for i in order:
            if len(picked) >= hi or (paid >= sel["remainEnergyCost"] and len(picked) >= lo):
                break
            picked.append(i)
            paid += opts[i].get("count") or 1
        return picked

    if context in BAD_FOR_OWNER:
        # Hurting the owner: hit opponent's stuff as much as allowed, our own stuff as little as allowed.
        theirs = [i for i in range(n) if opts[i].get("playerIndex", ctx.me) != ctx.me]
        if theirs and context in (C_DAMAGE_COUNTER, C_DAMAGE_COUNTER_ANY, C_DAMAGE):
            order = sorted(range(n), key=lambda i: -scores[i])
            k = hi
        elif theirs:
            order = sorted(range(n), key=lambda i: scores[i])  # their most valuable has the lowest score
            k = hi if len(theirs) == n else lo
        else:
            order = sorted(range(n), key=lambda i: scores[i])  # my least valuable first
            k = lo
        return order[:max(lo, min(k, hi))]

    # Good-for-me selections: take as many as allowed, best first.
    order = sorted(range(n), key=lambda i: -scores[i])
    k = hi
    if context in (C_SWITCH, C_TO_ACTIVE, C_SETUP_ACTIVE, C_ATTACK, C_DISABLE_ATTACK, C_RECOVER_SPECIAL_CONDITION):
        k = max(lo, 1) if hi >= 1 else 0
    return order[:max(lo, min(k, hi))]


def policy(obs_dict: dict) -> list[int]:
    """Rule-based choice (also used by both players inside search rollouts).

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    """
    sel = obs_dict["select"]
    n = len(sel["option"])
    try:
        picked = choose(Ctx(obs_dict))
        picked = list(dict.fromkeys(i for i in picked if 0 <= i < n))
        if sel["minCount"] <= len(picked) <= sel["maxCount"]:
            return picked
    except Exception:
        if DEBUG:
            raise
    # Fallback: never crash, always return something legal.
    return random.sample(range(n), sel["maxCount"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
DEBUG = bool(os.environ.get("AGENT_DEBUG"))
SEARCH = os.environ.get("AGENT_SEARCH", "1") != "0"
# Seconds of search per decision when the environment doesn't report remaining time (local testing).
LOCAL_BUDGET = float(os.environ.get("AGENT_BUDGET", "0.3"))
TOTAL_BUDGET = 600.0       # Kaggle overage time per agent for a whole Bo3 episode
SAFETY = 120.0             # never let remaining time drop below this
MAX_PER_DECISION = 2.0
MAIN_ONLY = os.environ.get("AGENT_MAIN_ONLY") == "1"
REAL_ME = None             # my player index in the real game (rollouts also play the opponent)

_searcher = None


def _search_budget(obs_dict) -> float:
    remaining = obs_dict.get("remainingOverageTime")
    if remaining is None:
        return LOCAL_BUDGET
    # ~40 searched decisions per game, up to 3 games; spend proportionally to what's left.
    return max(0.0, min(MAX_PER_DECISION, (remaining - SAFETY) / 80))


class _Globals:
    """Live view of this file's globals (Kaggle may exec this file without registering it as a module)."""

    def __getattr__(self, name):
        return globals()[name]


def _make_searcher():
    agent_dir = "/kaggle_simulations/agent"
    if os.path.isdir(agent_dir) and agent_dir not in sys.path:
        sys.path.append(agent_dir)
    import search
    return search.Searcher(_Globals())


def agent(obs_dict: dict) -> list[int]:
    global REAL_ME, _searcher
    if obs_dict.get("select") is None:
        # Initial selection: return the 60-card deck.
        return read_deck_csv()

    sel = obs_dict["select"]
    cur = obs_dict["current"]
    REAL_ME = cur["yourIndex"]
    fallback = policy(obs_dict)
    if not SEARCH:
        return fallback
    try:
        if _searcher is None:
            _searcher = _make_searcher()
        _searcher.opp_model.observe(obs_dict, 1 - REAL_ME)
        if cur["turn"] == 0 or len(sel["option"]) < 2 or not obs_dict.get("search_begin_input") \
                or sel["option"][0]["type"] == O_SKILL or (MAIN_ONLY and sel["context"] != C_MAIN):
            return fallback
        budget = _search_budget(obs_dict)
        if budget < 0.02:
            return fallback
        best = _searcher.choose(obs_dict, budget, fallback)
        if best is not None:
            return best
    except Exception:
        if DEBUG:
            raise
    return fallback
