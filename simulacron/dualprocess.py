"""Two ways of deciding, and when each one runs.

Until now every unit ran the full option-scoring loop every waking hour of its
life: enumerate everywhere it could go, price each one, weigh memory and fear
and obligation against all of them, choose. That is not how anybody lives.
Most of what a person does in a day is not decided at all -- it is the thing
they did last time in this situation, fired off without deliberation, and the
slow effortful machinery only engages when the fast path is out of its depth.

So: System 1 is a habit table keyed on the situation a unit finds itself in.
It is cheap, it is confident in proportion to how often it has worked, and it
runs by default. System 2 is the deliberation that was here already, and it
takes over when

  * the habit has no entry for this situation, or a weak one
  * something is badly wrong -- peril, hunger, a frightened or furious unit
  * the last prediction was violated, so the fast path has just been shown up
  * this is somewhere new

and it is *effortful*. Every engagement spends a budget that refills with rest
and sleep. A unit that has been deliberating all day gets worse at it and falls
back on habit, which is decision fatigue, and which is worth having because it
makes bad days compound the way they actually do.
"""

from dataclasses import dataclass, field


@dataclass
class Habit:
    intent: str
    venue: str = ""
    strength: float = 0.0      # how often this has been the answer here
    uses: int = 0
    payoff: float = 0.0        # running average of how it went


def situation(u, view):
    """The key a habit is stored under: the coarse shape of right now.

    Deliberately lossy. A habit that only fired in exactly identical
    circumstances would never fire at all.
    """
    if u.hunger > 0.7:
        need = "hungry"
    elif u.fatigue > 0.7:
        need = "tired"
    elif u.loneliness > 0.7:
        need = "alone"
    elif u.funds < 3 * 6.0:
        need = "broke"
    else:
        need = "fine"
    part = ("night" if view.hour < 6 or view.hour >= 22 else
            "morning" if view.hour < 12 else
            "afternoon" if view.hour < 18 else "evening")
    # Workday was in this key and made it twice as sparse for no behavioural
    # gain: what a workday changes is which options exist, which the slow path
    # sees anyway. Habits need to be coarse enough to actually fire.
    return f"{need}|{part}|{u.district}"


class Habits:
    def __init__(self):
        self.table = {}

    def get(self, key):
        return self.table.get(key)

    def reinforce(self, key, intent, venue, payoff, baseline=0.0):
        """Habits form from things that worked, not from things that happened.

        Reinforcing on mere repetition cemented whatever a unit did first in
        a situation, including going hungry, and the population got worse at
        feeding itself the moment the fast path took over.
        """
        if payoff < baseline:
            h = self.table.get(key)
            if h is not None:
                h.strength = max(0.0, h.strength - 0.10)
                h.payoff += 0.25 * (payoff - h.payoff)
            return h
        h = self.table.get(key)
        if h is None or h.intent != intent or h.venue != venue:
            if h is not None and h.strength > 0.25:
                h.strength -= 0.12        # a competing answer weakens the old one
                return h
            h = self.table[key] = Habit(intent=intent, venue=venue)
        h.uses += 1
        h.strength = min(1.0, h.strength + 0.20)
        h.payoff += 0.25 * (payoff - h.payoff)
        return h

    def confidence(self, key):
        h = self.table.get(key)
        if h is None:
            return 0.0
        # a habit that has been working is trusted; one that has not is not
        return h.strength * (0.35 + 0.65 * max(0.0, min(1.0, h.payoff)))

    def size(self):
        return len(self.table)


@dataclass
class Effort:
    """The budget System 2 spends, and decision fatigue as a consequence."""

    pool: float = 1.0

    def spend(self, amount=0.055):
        self.pool = max(0.0, self.pool - amount)

    def recover(self, amount):
        self.pool = min(1.0, self.pool + amount)

    def available(self):
        return self.pool


def needs_thought(u, view, key):
    """Whether the fast path should hand over, and why.

    Returns a reason string, or "" to let habit run. The reason is kept so a
    unit's own account of why it stopped to think can be read back.
    """
    # These thresholds matter more than they look. Set where they first felt
    # reasonable, three quarters of all decisions went to the slow path and
    # the dual-process split did nothing at all -- reward prediction error in
    # particular swings widely from hour to hour and was flagging almost every
    # outcome as a surprise worth stopping for.
    if u.peril() > 0.30:
        return "in danger"
    if u.surprise > 0.95:
        return "that did not go as expected"
    if u.affect.intensity["fear"] > 0.45 or u.affect.intensity["anger"] > 0.5:
        return "worked up"
    if u.habits.confidence(key) < 0.22:
        return "no idea what to do here"
    if view.here is not None and view.here.visits <= 1:
        return "somewhere new"
    if u.dependents_worry() > 0.55:
        return "worried about home"
    if u.rng.random() < 0.04:
        return "no particular reason"       # people stop and think sometimes
    return ""
