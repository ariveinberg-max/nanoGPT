"""Who a unit is, as distinct from what it does.

The five traits a unit used to carry -- sociability, diligence, appetite,
restlessness, thrift -- were invented for the jobs the code needed doing. They
are not a personality; they are five knobs named after their effects. This is
the Big Five instead, on the one-to-ten scale people actually use, with those
behavioural knobs derived from it so nothing downstream has to change.

Two more things a person has that a set of drives does not.

**Values.** What somebody cares about most, which is not the same as what they
need. Two units equally hungry and equally broke will still choose differently
if one of them cares about family and the other about getting out. Values are
tiebreakers, and they are what makes a unit's choices legible as *that unit's*
choices rather than as the output of a scoring function everybody shares.

**Interests.** Things pursued for no reason at all. This is the part that
stops a population being a set of job-holders: a unit that loves the ocean
goes to Venice when it is not hungry, not lonely and not due anywhere, because
it likes the ocean. Nothing in the drive system can produce that, and a city
where nobody does it does not look like a city.
"""

from dataclasses import dataclass, field

BIG_FIVE = ("openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism")

SHORT = {"openness": "O", "conscientiousness": "C", "extraversion": "E",
         "agreeableness": "A", "neuroticism": "N"}

VALUES = ("security", "family", "freedom", "standing", "belonging",
          "craft", "pleasure", "fairness")

# What a unit is drawn to, and where that takes it. The venue keys are the
# ones in world.py; an interest is a pull toward a place, not a need met by it.
INTERESTS = {
    "the ocean": ("boardwalk", "sm_beach", "plunge"),
    "pictures": ("arclight", "graumans"),
    "music": ("world_stage", "the_echo", "club_alabam", "noraebang"),
    "books": ("central_library",),
    "the sky": ("observatory",),
    "drink": ("la_live", "the_echo", "noraebang"),
    "walking": ("echo_park", "mariachi_plaza", "leimert_plaza", "pershing_square"),
    "eating well": ("ktown_bbq", "musso", "leimert_soul", "philippes"),
}


@dataclass
class Person:
    """Personality, what it cares about, and what it is drawn to."""

    five: dict = field(default_factory=dict)      # 1-10 each
    values: tuple = ()                            # ranked, most-held first
    interests: tuple = ()
    quirk: str = ""

    # -- the derived knobs the rest of the code already uses ---------------
    def traits(self, rng):
        """The behavioural traits, read off the personality.

        Kept as the same five names so nothing downstream changes, but they
        are now consequences of a personality rather than the whole of one.
        """
        f = self.five
        n = lambda k: (f[k] - 1) / 9.0           # noqa: E731  -> 0..1
        jitter = lambda: float(rng.normal(0, 0.06))  # noqa: E731
        clamp = lambda x: float(max(0.05, min(1.0, x)))  # noqa: E731
        return {
            "sociability": clamp(0.15 + 0.8 * n("extraversion") + jitter()),
            "diligence": clamp(0.15 + 0.8 * n("conscientiousness") + jitter()),
            "restlessness": clamp(0.1 + 0.55 * n("openness")
                                  + 0.35 * (1 - n("conscientiousness")) + jitter()),
            "thrift": clamp(0.15 + 0.6 * n("conscientiousness")
                            + 0.25 * (1 - n("openness")) + jitter()),
            # appetite is a body, not a personality
            "appetite": clamp(0.6 + 0.8 * float(rng.random())),
        }

    def scale(self, key):
        """0..1 view of one factor, for code that wants a weight."""
        return (self.five.get(key, 5) - 1) / 9.0

    def holds(self, value):
        """How much this unit cares about something, 0 if not at all."""
        if value not in self.values:
            return 0.0
        rank = self.values.index(value)
        return 1.0 - 0.32 * rank

    def describe(self):
        five = " ".join(f"{SHORT[k]}{self.five[k]}" for k in BIG_FIVE)
        return (f"{five} · cares about {', '.join(self.values)} · "
                f"{', '.join(self.interests) or 'nothing in particular'}"
                + (f" · {self.quirk}" if self.quirk else ""))

    def sketch(self):
        """A sentence someone might actually say about this person."""
        f, bits = self.five, []
        bits.append("outgoing" if f["extraversion"] >= 7 else
                    "keeps to themselves" if f["extraversion"] <= 3 else
                    "sociable enough")
        if f["conscientiousness"] >= 7:
            bits.append("reliable")
        elif f["conscientiousness"] <= 3:
            bits.append("does not keep to much")
        if f["neuroticism"] >= 7:
            bits.append("takes things hard")
        elif f["neuroticism"] <= 3:
            bits.append("hard to rattle")
        if f["openness"] >= 7:
            bits.append("curious")
        if f["agreeableness"] <= 3:
            bits.append("prickly")
        elif f["agreeableness"] >= 8:
            bits.append("easy to get on with")
        return ", ".join(bits)


QUIRKS = (
    "walks everywhere rather than take the train",
    "cannot sit with their back to a door",
    "always early, and resents everyone who is not",
    "talks to people they have not been introduced to",
    "will not eat anywhere they have not eaten before",
    "keeps every receipt",
    "counts things without meaning to",
    "goes quiet when tired rather than saying so",
    "never finishes a drink",
    "apologises when other people are wrong",
)


def make(rng):
    """Draw a person. Traits are correlated the way they are in people --
    not independently, or everybody comes out average on everything."""
    base = float(rng.normal(0, 1))
    def factor(bias=0.0, load=0.0):
        raw = 5.5 + 2.1 * float(rng.normal(0, 1)) + load * base + bias
        return int(max(1, min(10, round(raw))))

    five = {
        "openness": factor(load=0.5),
        "conscientiousness": factor(load=-0.2),
        "extraversion": factor(load=0.6),
        "agreeableness": factor(load=0.2),
        "neuroticism": factor(load=-0.4),
    }
    n_values = int(rng.integers(2, 4))
    values = tuple(rng.choice(VALUES, size=n_values, replace=False))
    n_int = int(rng.integers(1, 4))
    interests = tuple(rng.choice(list(INTERESTS), size=n_int, replace=False))
    quirk = str(rng.choice(QUIRKS)) if rng.random() < 0.55 else ""
    return Person(five=five, values=values, interests=interests, quirk=quirk)


# One thing that already happened before the prototype opened. A life that
# starts blank is not a life anybody recognises: people arrive carrying
# something. Tied to what the unit cares about, because what you care about
# and what happened to you are not independent.
FORMATIVE = {
    "security": ("the house went that winter, and we moved twice after",
                 {"fear": 0.55, "sadness": 0.7}),
    "family": ("was left at a relative's one summer and never told why",
               {"sadness": 0.65, "shame": 0.35}),
    "freedom": ("got out of somewhere once, and has never quite settled since",
                {"relief": 0.5, "hope": 0.45}),
    "standing": ("was laughed at, in front of the people who mattered",
                 {"shame": 0.75, "anger": 0.4}),
    "belonging": ("was the only one not asked, and understood why",
                  {"sadness": 0.6, "shame": 0.5}),
    "craft": ("made something once that people stopped to look at",
              {"pride": 0.75, "joy": 0.4}),
    "pleasure": ("one summer that everything since has been measured against",
                 {"joy": 0.7, "sadness": 0.3}),
    "fairness": ("watched somebody get away with it, and nobody said anything",
                 {"anger": 0.7, "sadness": 0.4}),
}


def formative(who, rng):
    """The thing this unit arrived carrying."""
    if not who.values:
        return None
    key = str(rng.choice(who.values))
    return FORMATIVE.get(key)


def draw_to(person):
    """Venue keys this unit is drawn to for their own sake."""
    out = set()
    for i in person.interests:
        out.update(INTERESTS.get(i, ()))
    return out
