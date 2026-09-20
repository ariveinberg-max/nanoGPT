"""What a unit is trying to bring about.

Units had drives, which is not the same thing. A drive is a pressure that
rises until you deal with it; a goal is something you are *for*, that persists
across days when nothing is pressing, and that makes one hour's choice legible
in terms of another's. Without them a unit is only ever reacting, and a life
made entirely of reactions does not look like a life.

Two horizons. A week's intention comes out of circumstance -- the debt, the
lack of work, not having seen anyone -- and gets dropped or met fairly soon.
An ambition comes out of what a unit cares about, lasts, and mostly does not
get met at all, which is also true of ambitions.
"""

from dataclasses import dataclass


@dataclass
class Goal:
    kind: str
    horizon: str            # "week" | "life"
    said: str               # how the unit would put it
    target: str = ""        # a person or a district, where it has one
    formed: int = 0
    progress: float = 0.0
    done: bool = False
    given_up: bool = False

    def alive(self):
        return not (self.done or self.given_up)


def week_goals(u, tick, cost):
    """The intentions this week's circumstances suggest."""
    out = []
    if u.debt > 3 * cost:
        out.append(Goal("clear_debt", "week", "get clear of what I owe", formed=tick))
    if u.status == "unemployed":
        out.append(Goal("find_work", "week", "find something", formed=tick))
    if u.loneliness > 0.6:
        close = u.social.closest(1)
        who = close[0].name if close else ""
        out.append(Goal("see_someone", "week",
                        f"see {who}" if who else "see somebody",
                        target=who, formed=tick))
    if u.health < 0.6 or u.pain > 0.4:
        out.append(Goal("get_well", "week", "get back on my feet", formed=tick))
    if u.hunger > 0.8 and u.funds < cost:
        out.append(Goal("eat_properly", "week", "eat properly for once", formed=tick))
    return out


def life_goals(u, tick):
    """What a unit is for, drawn from what it cares about."""
    who = u.person
    if who is None:
        return []
    out = []
    if who.holds("family") > 0.5 and not u.family.children:
        out.append(Goal("family", "life", "have people of my own", formed=tick))
    if who.holds("security") > 0.5:
        out.append(Goal("put_by", "life", "put something by", formed=tick))
    if who.holds("standing") > 0.5:
        out.append(Goal("make_good", "life", "make something of myself", formed=tick))
    if who.holds("freedom") > 0.5:
        out.append(Goal("see_it", "life", "see something of the world", formed=tick))
    if who.holds("craft") > 0.5:
        out.append(Goal("be_good_at_it", "life", "be good at what I do", formed=tick))
    if who.holds("belonging") > 0.5:
        out.append(Goal("belong", "life", "have somewhere I belong", formed=tick))
    return out[:2]


def review(u, tick, cost, by_name):
    """Met, given up on, or still going. Returns (met, dropped)."""
    met, dropped = [], []
    for g in list(u.goals):
        if not g.alive():
            continue
        if g.kind == "clear_debt":
            g.progress = 1.0 - min(1.0, u.debt / max(cost, 1.0) / 3.0)
            if u.debt <= 0.01:
                g.done = True; met.append(g)
        elif g.kind == "find_work":
            if u.status == "employed":
                g.done = True; met.append(g)
            elif u.status not in ("unemployed",):
                g.given_up = True; dropped.append(g)
        elif g.kind == "see_someone":
            rel = u.social.people.get(g.target) if g.target else None
            if rel and tick - rel.last_seen < 96:
                g.done = True; met.append(g)
        elif g.kind == "get_well":
            if u.health > 0.85 and u.pain < 0.15:
                g.done = True; met.append(g)
        elif g.kind == "eat_properly":
            if u.hunger < 0.3:
                g.done = True; met.append(g)
        elif g.kind == "family":
            if u.family.children:
                g.done = True; met.append(g)
        elif g.kind == "put_by":
            g.progress = min(1.0, u.funds / (30.0 * cost))
            if g.progress >= 1.0:
                g.done = True; met.append(g)
        elif g.kind == "make_good":
            g.progress = u.selfmodel.esteem()
        elif g.kind == "belong":
            g.progress = min(1.0, u.social.known() / 6.0)
        elif g.kind == "see_it":
            g.progress = min(1.0, len(u.memory.places) / 8.0)

        # a week's intention that has gone nowhere for long enough is dropped
        if g.alive() and g.horizon == "week" and tick - g.formed > 96 * 21:
            g.given_up = True
            dropped.append(g)
    u.goals = [g for g in u.goals if g.alive() or tick - g.formed < 96 * 3]
    return met, dropped


def pursuing(u, kind):
    return any(g.kind == kind and g.alive() for g in u.goals)


def toward(u, opt):
    """How much this option serves what the unit is trying to bring about."""
    w = 0.0
    for g in u.goals:
        if not g.alive():
            continue
        if g.kind in ("clear_debt", "put_by") and opt.intent == "work":
            w += 0.75
        elif g.kind in ("clear_debt",) and opt.intent in ("socialize", "errand"):
            w -= 0.45
        elif g.kind == "find_work" and opt.intent in ("errand", "seek"):
            w += 0.7
        elif g.kind == "see_someone" and opt.person and opt.person == g.target:
            w += 1.0
        elif g.kind in ("get_well",) and opt.intent in ("rest", "sleep", "eat"):
            w += 0.55
        elif g.kind == "eat_properly" and opt.intent == "eat":
            w += 0.7
        elif g.kind in ("family", "belong") and opt.intent == "socialize":
            w += 0.5
        elif g.kind == "see_it" and opt.intent in ("wander", "seek"):
            w += 0.6
        elif g.kind == "be_good_at_it" and opt.intent == "work":
            w += 0.6
        elif g.kind == "make_good" and opt.intent == "work":
            w += 0.45
    return w


def describe(u):
    live = [g for g in u.goals if g.alive()]
    if not live:
        return "nothing in particular"
    week = [g.said for g in live if g.horizon == "week"]
    life = [g.said for g in live if g.horizon == "life"]
    bits = []
    if week:
        bits.append("this week: " + "; ".join(week))
    if life:
        bits.append("in the end: " + "; ".join(life))
    return " · ".join(bits)
