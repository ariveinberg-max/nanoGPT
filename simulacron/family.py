"""Kinship, birth, and belief that outlives the believer.

Everything a unit knew used to die with it. A unit that learned over twenty
years which streets to avoid took that with it and the next arrival started
from nothing, so the population never accumulated anything. Nobody in the
prototype had ever been told something.

Children here are not spawned. They are born to particular people, in a
particular flat, and they begin with their parents' account of the world
rather than with a blank one -- which streets are dangerous, who can be
trusted, what a person is for, and how much of it to doubt. None of that is
learned from experience, because a child has none. It is inherited, and it is
wrong in the specific ways its parents were wrong.

That is where culture comes from, and it is the only mechanism here that
produces something no one wrote.
"""

from dataclasses import dataclass, field

from . import lifecourse

PAIR_MIN_CLOSENESS = 0.45
FERTILE_FROM, FERTILE_TO = 20.0, 42.0
DEPENDENT_UNTIL = 14.0


@dataclass
class Family:
    partner: str = ""
    parents: tuple = ()
    children: list = field(default_factory=list)

    def dependents(self, by_name):
        out = []
        for name in self.children:
            child = by_name.get(name)
            if child is not None and child.age < DEPENDENT_UNTIL:
                out.append(child)
        return out

    def kin(self):
        return set(self.children) | set(self.parents) | (
            {self.partner} if self.partner else set())

    def describe(self):
        bits = []
        if self.partner:
            bits.append(f"with {self.partner}")
        if self.children:
            bits.append(f"{len(self.children)} "
                        + ("child" if len(self.children) == 1 else "children"))
        if self.parents:
            bits.append(f"child of {' and '.join(self.parents)}")
        return "; ".join(bits) or "no one"


def may_pair(a, b):
    if a.family.partner or b.family.partner:
        return False
    if a.age < lifecourse.ADULT or b.age < lifecourse.ADULT:
        return False
    if b.name in a.family.kin() or a.name in b.family.kin():
        return False
    return (a.social.of(b.name).closeness() > PAIR_MIN_CLOSENESS
            and b.social.of(a.name).closeness() > PAIR_MIN_CLOSENESS)


def may_bear(a, b):
    return (a.family.partner == b.name
            and all(FERTILE_FROM <= u.age <= FERTILE_TO for u in (a, b))
            and all(u.health > 0.6 for u in (a, b))
            and len(a.family.children) < 4)


def inherit_beliefs(child, parents):
    """A child starts with its parents' account of things, not with none.

    Place beliefs, trust priors and the endorsed shape of a person all come
    across. So does doubt: a child raised by someone who had stopped believing
    the world was the shape they were told does not start from certainty.
    """
    from .memory import PlaceBelief
    from .worldview import inherit

    primary = max(parents, key=lambda p: p.worldview.confidence)
    shared = sum(p.worldview.confidence for p in parents) / len(parents)
    child.worldview = inherit(primary.worldview, confidence=shared)

    for parent in parents:
        # which streets are dangerous, taken on trust rather than on evidence
        for district, belief in parent.memory.places.items():
            mine = child.memory.places.setdefault(district, PlaceBelief())
            mine.danger = max(mine.danger, belief.danger * 0.7)
            mine.warmth = max(mine.warmth, belief.warmth * 0.5)
            mine.visits = max(mine.visits, 2)
        # and who your people think is worth trusting
        for name, rel in parent.social.people.items():
            if name == child.name or rel.familiarity < 0.3:
                continue
            mine = child.social.of(name)
            mine.trust = 0.5 + 0.6 * (rel.trust - 0.5)
            mine.affection = 0.5 * rel.affection
        # what a person is supposed to be
        for role, weight in parent.selfmodel.roles.items():
            child.selfmodel.roles[role] += 0.18 * weight / len(parents)

    for role in child.selfmodel.roles:
        child.selfmodel.roles[role] = min(1.0, child.selfmodel.roles[role])
    return child


def blend_traits(a, b, rng):
    """Children resemble their parents, imperfectly."""
    out = {}
    for k in a.traits:
        mid = 0.5 * (a.traits[k] + b.traits[k])
        out[k] = float(max(0.05, mid + rng.normal(0, 0.12)))
    return out
