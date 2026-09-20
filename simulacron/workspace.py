"""Attention, and a unit's representation of its own states.

Everything in this simulation has so far been available to everything else at
once. Every drive, every memory, every percept fed the same decision with
equal standing, which is not how a mind is organised and is exactly the thing
Global Workspace Theory is a theory about: most processing stays local and
unconscious, and only what wins a competition gets broadcast widely enough to
reach memory, deliberation and report together.

So contents bid. A percept, a drive that has got loud, an intrusive memory, a
face across the room, a pain -- each arrives with a salience, one wins the
tick, and only the winner is broadcast: written to memory, offered to
deliberation, and available to be reported on. The rest happens and is gone.

The second half is the recursive part. The unit does not only have states, it
builds a representation *of* its states -- "I am afraid", not merely being
afraid -- and that representation is itself content that can be attended to,
stored, and used. It is also fallible, which is the point: a unit constructs
an account of why it did what it did, and the account can be wrong, because it
is assembled from what happened to be in the workspace rather than read off
the machinery that actually produced the behaviour.
"""

from dataclasses import dataclass, field

from .affect import ADJECTIVE

KINDS = ("percept", "drive", "memory", "social", "pain", "thought")


@dataclass
class Content:
    kind: str
    what: str                       # as the lab would put it
    salience: float
    detail: dict = field(default_factory=dict)
    mine: str = ""                  # as the unit would put it

    def said(self):
        return self.mine or self.what

    def __str__(self):
        return self.what


@dataclass
class Workspace:
    """One slot, contested. What is in it is what the unit is 'on'."""

    current: object = None
    last: object = None
    broadcasts: int = 0
    history: list = field(default_factory=list)      # recent winners, for report

    def compete(self, candidates):
        """Winner takes the slot. Everything else is processed and discarded."""
        if not candidates:
            self.current = None
            return None
        winner = max(candidates, key=lambda c: c.salience)
        self.last, self.current = self.current, winner
        self.broadcasts += 1
        self.history.append(winner)
        if len(self.history) > 24:
            del self.history[:12]
        return winner

    def attended(self, kind):
        return self.current is not None and self.current.kind == kind

    def recent(self, n=5):
        return self.history[-n:]


def bid(u, view):
    """Everything competing for this unit's attention this hour."""
    out = []
    a = u.affect

    for drive, level, word in (
            ("hunger", u.hunger, "how hungry it is"),
            ("fatigue", u.fatigue, "how tired it is"),
            ("loneliness", u.loneliness, "being on its own"),
    ):
        if level > 0.45:
            out.append(Content("drive", word, salience=level * 0.9,
                               detail={"drive": drive, "level": level},
                               mine={"hunger": "how hungry I am",
                                     "fatigue": "how tired I am",
                                     "loneliness": "being on my own"}[drive]))
    if u.pain > 0.2:
        out.append(Content("pain", "the pain", salience=0.55 + 0.45 * u.pain,
                           detail={"level": u.pain}, mine="the pain"))

    emotion, strength = a.dominant()
    if strength > 0.25:
        out.append(Content("thought", f"being {ADJECTIVE.get(emotion, emotion)}",
                           salience=strength,
                           detail={"emotion": emotion, "strength": strength},
                           mine=f"being {ADJECTIVE.get(emotion, emotion)}"))

    if u.memory.intrusion is not None:
        ep = u.memory.intrusion
        out.append(Content("memory", ep.text, salience=0.55 + 0.45 * ep.salience,
                           detail={"episode": ep}, mine=ep.text))

    if view is not None:
        for s in view.present:
            rel = u.social.people.get(s.name)
            weight = 0.3 + 0.5 * (rel.closeness() if rel else 0.0) \
                + 0.6 * (rel.wariness() if rel else 0.0)
            out.append(Content("social", f"{s.name} being here", salience=weight,
                               detail={"who": s.name},
                               mine=f"{s.name} being here"))
        if view.here is not None:
            danger = u.memory.danger_of(view.here.district)
            out.append(Content("percept", f"where it is -- {view.here.name}",
                               salience=0.25 + 0.7 * danger,
                               detail={"place": view.here.key},
                               mine=f"being at {view.here.name}"))
    return out


def self_report(u):
    """The unit's account of its own state. A representation of a representation.

    Assembled from what is in the workspace and what it believes about itself,
    not read off the machinery that is actually driving it -- which is why it
    can be wrong, and why `why_thought` and this can disagree.
    """
    a = u.affect
    emotion, strength = a.dominant()
    bits = []
    if u.workspace.current is not None:
        bits.append(f"I am thinking about {u.workspace.current.said()}")
    if strength > 0.2:
        bits.append(f"I am {ADJECTIVE.get(emotion, emotion)}")
    elif a.stress > 0.5:
        bits.append("I am worn down")
    else:
        bits.append("I am alright")
    if u.peril() > 0.25:
        bits.append("I am not sure I am going to be alright")
    if u.chose is not None:
        bits.append(f"I am going to {u.chose.label()}")
    return "; ".join(bits)


def confabulate(u):
    """Why the unit thinks it did that.

    It does not have access to the scoring. It has access to what was in the
    workspace at the time and to what it believes about itself, and it builds
    an explanation out of those. Sometimes that matches the reason the choice
    actually won. Often it does not, and the unit is not lying.
    """
    if u.chose is None:
        return "no reason"
    attended = u.workspace.current
    if attended is not None and attended.kind in ("drive", "pain"):
        return f"because of {attended.said()}"
    if attended is not None and attended.kind == "memory":
        return f"because I kept thinking about {attended.said()}"
    if attended is not None and attended.kind == "social":
        return f"because of {attended.said()}"
    role = u.selfmodel.identity(u.age)
    if role not in ("nobody in particular", "somebody's child"):
        return f"because that is what a {role} does"
    return "because it seemed like the thing to do"


def honest_reason(u):
    """What actually decided it, for comparison. The unit has no access to this."""
    if u.chose is None or not u.chose.reasons:
        return "nothing"
    k, v = max(u.chose.reasons.items(), key=lambda kv: abs(kv[1]))
    return f"{k} {v:+.2f}"
