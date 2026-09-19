"""Other units, as individuals.

In the previous build company was a number: how many bodies share this room.
It changed how fast loneliness drained and nothing else. A unit had no way to
prefer one person to another, to remember who had helped it, or to cross the
street when it saw someone coming.

Here every unit a unit has met has a standing in its head -- how well it knows
them, whether it trusts them, whether it likes them, what it is still holding
against them -- and those standings are what the decision loop reads when it
decides whose company is worth the fare across town.
"""

from dataclasses import dataclass, field


@dataclass
class Relationship:
    name: str
    familiarity: float = 0.0        # 0 a stranger .. 1 knows them well
    trust: float = 0.5              # 0 would not turn their back .. 1 relies on them
    affection: float = 0.0          # -1 cannot stand them .. 1 loves them
    grudge: float = 0.0             # what is still owed, and forgiven slowly
    encounters: int = 0
    last_seen: int = -10 ** 9

    def closeness(self):
        return max(0.0, self.familiarity * (0.4 + 0.6 * (self.affection + 1) / 2)
                   - 0.5 * self.grudge)

    def wariness(self):
        return max(0.0, (1.0 - self.trust) * 0.5 + self.grudge)

    def describe(self):
        if self.grudge > 0.35:
            tone = "holds it against them"
        elif self.affection > 0.4:
            tone = "fond of them"
        elif self.affection < -0.3:
            tone = "dislikes them"
        elif self.familiarity > 0.5:
            tone = "knows them"
        else:
            tone = "barely knows them"
        return (f"{self.name}: {tone} (familiarity {self.familiarity:.2f}, "
                f"trust {self.trust:.2f}, grudge {self.grudge:.2f})")


class Social:
    def __init__(self):
        self.people = {}

    def of(self, name):
        r = self.people.get(name)
        if r is None:
            r = self.people[name] = Relationship(name)
        return r

    def met(self, name, tick, quality=0.0):
        """Time spent together. Familiarity is cheap; affection is not."""
        r = self.of(name)
        r.encounters += 1
        r.last_seen = tick
        r.familiarity = min(1.0, r.familiarity + 0.06 * (1.0 - r.familiarity))
        r.affection = max(-1.0, min(1.0, r.affection + 0.04 * quality))
        return r

    def treated(self, name, tick, valence, betrayal=0.0):
        """Someone did something to this unit. This is where trust moves."""
        r = self.of(name)
        r.last_seen = tick
        r.trust = max(0.0, min(1.0, r.trust + 0.12 * valence - 0.35 * betrayal))
        r.affection = max(-1.0, min(1.0, r.affection + 0.10 * valence))
        if valence < 0 or betrayal:
            r.grudge = min(1.0, r.grudge + 0.30 * (betrayal + max(0.0, -valence)))
        return r

    def heard(self, name, valence, credibility):
        """Second-hand. Cheaper to acquire and cheaper to discard."""
        r = self.of(name)
        r.trust = max(0.0, min(1.0, r.trust + 0.05 * valence * credibility))
        r.affection = max(-1.0, min(1.0, r.affection + 0.03 * valence * credibility))

    def forget(self):
        """Grudges soften. Familiarity with people you never see fades."""
        for r in self.people.values():
            r.grudge = max(0.0, r.grudge - 0.0008)
            r.familiarity = max(0.0, r.familiarity - 0.0004)

    # -- readouts -----------------------------------------------------------
    def closest(self, n=3):
        return sorted(self.people.values(), key=lambda r: -r.closeness())[:n]

    def feared(self, n=3):
        return [r for r in sorted(self.people.values(), key=lambda r: -r.wariness())[:n]
                if r.wariness() > 0.3]

    def known(self):
        return sum(1 for r in self.people.values() if r.familiarity > 0.2)

    def describe(self, n=4):
        rows = sorted(self.people.values(), key=lambda r: -r.familiarity)[:n]
        return [r.describe() for r in rows]
