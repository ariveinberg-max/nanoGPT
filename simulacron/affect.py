"""Emotion, derived rather than declared.

The old build had one number called `mood`. This one has no emotion variables
that anything sets directly. Instead an event is *appraised* -- was it good or
bad for what I wanted, has it happened or might it, did someone do it to me or
did I do it, can I do anything about it, was a rule broken -- and the emotions
fall out of that pattern. Being robbed and being laid off are both bad, but one
has an agent and the other does not, so one produces anger and the other
produces fear and shame. That distinction is the whole point of doing it this
way, and it is not available to a single mood scalar.

The scheme is a pared-down OCC/Scherer appraisal model: enough structure to
separate the emotions that matter behaviourally, not a theory of feeling.
"""

from dataclasses import dataclass, field

EMOTIONS = ("fear", "anger", "sadness", "shame", "guilt",
            "joy", "pride", "hope", "relief")

# How fast each emotion fades per tick. Fear and anger are hot and short;
# sadness and shame are cold and long.
DECAY = {
    "fear": 0.045,
    "anger": 0.055,
    "sadness": 0.012,
    "shame": 0.015,
    "guilt": 0.014,
    "joy": 0.060,
    "pride": 0.030,
    "hope": 0.025,
    "relief": 0.080,
}

# Emotions that push a unit towards or away from effort and company.
# How a unit would say it, as opposed to how the system stores it.
ADJECTIVE = {
    "fear": "afraid", "anger": "angry", "sadness": "sad", "shame": "ashamed",
    "guilt": "guilty", "joy": "glad", "pride": "proud", "hope": "hopeful",
    "relief": "relieved", "settled": "alright",
}

NEGATIVE = ("fear", "anger", "sadness", "shame", "guilt")
POSITIVE = ("joy", "pride", "hope", "relief")
WITHDRAWING = ("sadness", "shame", "guilt", "fear")
ENERGISING = ("anger", "hope", "joy", "pride")


@dataclass
class Appraisal:
    """One event, scored on the dimensions emotions are made of."""

    valence: float                     # -1 harmful .. +1 beneficial, to *my* goals
    certainty: float = 1.0             # 0 it might happen .. 1 it has happened
    anticipated: bool = False          # is this about the future?
    agency: str = "circumstance"       # self | other | circumstance
    control: float = 0.5               # how much I can do about it
    norm: float = 0.0                  # -1 a rule was broken .. +1 upheld
    irreversible: float = 0.0          # 0 recoverable .. 1 a loss that stays lost
    public: float = 0.0                # 0 nobody saw .. 1 it was witnessed
    harmed_other: float = 0.0          # how much my action cost someone else
    averted: bool = False              # a feared thing did not come to pass
    other: str = ""                    # whose doing, or who it happened to

    def emotions(self):
        """Derive emotion intensities from the appraisal pattern."""
        v, c = self.valence, self.certainty
        bad, good = max(0.0, -v), max(0.0, v)
        helpless = 1.0 - self.control
        out = {e: 0.0 for e in EMOTIONS}

        if self.anticipated:
            # prospect-based: what might be coming
            uncertainty = 1.0 - abs(2 * c - 1)
            out["fear"] = bad * helpless * (0.4 + 0.6 * uncertainty)
            out["hope"] = good * (0.4 + 0.6 * uncertainty)
        else:
            # event-based: what has happened
            if self.averted:
                out["relief"] = max(good, bad) * c
            if self.agency == "other":
                # Anger keys off blame, not off whether I could have stopped
                # it -- being robbed while helpless is not less enraging. What
                # my control changes is whether the anger has anywhere to go,
                # and how much of the event lands as fear instead.
                blame = 0.4 + 0.6 * max(0.0, -self.norm)
                out["anger"] = bad * c * blame * (0.55 + 0.45 * self.control)
                out["fear"] = bad * c * helpless * 0.5
                out["sadness"] = bad * c * helpless * (0.3 + 0.7 * self.irreversible)
                out["joy"] = good * c * 0.6
            elif self.agency == "self":
                out["pride"] = good * c * (0.5 + 0.5 * max(0.0, self.norm))
                out["shame"] = bad * c * max(0.0, -self.norm) * (0.3 + 0.7 * self.public)
                out["guilt"] = self.harmed_other * c
                out["sadness"] = bad * c * helpless * 0.5
            else:
                out["sadness"] = bad * c * (0.4 + 0.6 * self.irreversible)
                out["fear"] = bad * c * helpless * 0.4
                out["joy"] = good * c

        return {e: min(1.0, round(x, 6)) for e, x in out.items() if x > 1e-6}


@dataclass
class Affect:
    """A unit's current feeling, and what a long run of feeling does to it."""

    intensity: dict = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})
    stress: float = 0.0            # chronic load; slow up, slower down
    trauma: float = 0.0            # what the worst episodes leave behind

    reactivity: float = 1.0        # how hard things land, from neuroticism

    def feel(self, emotions, scale=1.0):
        """Add an appraisal's emotions to the current state.

        Scaled by reactivity, so the same event does not land the same way on
        everybody -- which is most of what it means to say two people have
        different temperaments.
        """
        for e, x in emotions.items():
            gain = self.reactivity if e in NEGATIVE else \
                (2.0 - self.reactivity)
            self.intensity[e] = min(1.0, self.intensity[e] + x * scale * gain)
        hit = sum(self.intensity[e] for e in ("fear", "anger", "sadness", "shame", "guilt"))
        if hit > 1.4:
            self.trauma = min(1.0, self.trauma + 0.004 * (hit - 1.4))

    def decay(self):
        for e in EMOTIONS:
            self.intensity[e] = max(0.0, self.intensity[e] - DECAY[e])
        # Stress is driven by the raw weight of what a unit is carrying, not
        # by the per-emotion average: averaging over nine slots made a unit
        # that was frightened and ashamed look almost settled, and chronic
        # stress could never accumulate at all.
        load = self._sum(NEGATIVE) - 0.4 * self._sum(POSITIVE)
        self.stress = min(1.0, max(0.0, self.stress + 0.015 * load * self.reactivity
                                  - 0.0018 * (2.0 - self.reactivity)))
        self.trauma = max(0.0, self.trauma - 0.00002)

    def _sum(self, names):
        return sum(self.intensity[e] for e in names)

    # -- readouts -----------------------------------------------------------
    def negative(self):
        return self._sum(NEGATIVE) / len(NEGATIVE)

    def positive(self):
        return self._sum(POSITIVE) / len(POSITIVE)

    def valence(self):
        return self.positive() - self.negative()

    def arousal(self):
        return min(1.0, self.intensity["fear"] + self.intensity["anger"]
                   + 0.5 * self.intensity["joy"] + 0.3 * self.stress)

    def drive(self):
        """Willingness to spend effort on anything. Despair is not laziness."""
        pull = sum(self.intensity[e] for e in ENERGISING)
        drag = sum(self.intensity[e] for e in WITHDRAWING)
        return max(0.15, 1.0 + 0.25 * pull - 0.45 * drag - 0.5 * self.stress)

    def dominant(self):
        e, x = max(self.intensity.items(), key=lambda kv: kv[1])
        return (e, x) if x > 0.05 else ("settled", 0.0)

    def describe(self):
        live = sorted(((x, e) for e, x in self.intensity.items() if x > 0.05),
                      reverse=True)
        if not live:
            mood = "settled"
        else:
            mood = ", ".join(f"{e} {x:.2f}" for x, e in live[:3])
        return (f"{mood} | stress {self.stress:.2f} | trauma {self.trauma:.2f} "
                f"| drive {self.drive():.2f}")
