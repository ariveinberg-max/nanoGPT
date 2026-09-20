"""What a unit knows about a world that was never built.

The prototype is eight districts. There is no California around it, no ocean
floor, no other hemisphere, no moon at two hundred and forty thousand miles.
There is a rendered volume and an edge.

But a person is mostly made of things they have never checked. You believe in
Lisbon. You have a rough, confident account of the war, the ocean, the sky.
Almost none of it is yours by experience; it was handed to you and you never
had cause to test it. So each unit carries the same: received knowledge of an
Earth it can neither reach nor verify, held with exactly the confidence people
hold such things.

The interesting part is not the beliefs. It is the seam. A unit that goes
looking -- walks to the edge of town, asks what lies past the water, stands at
Griffith Observatory and looks up -- is querying a world model against a world
that does not extend that far, and it gets back something that does not
resolve. That mismatch is what `dissonance` measures. It was a bare counter
before, incremented for standing in Venice. Now it is what it says it is:
evidence the world is not shaped the way the unit was told.
"""

from dataclasses import dataclass, field

# Received geography. None of this is simulated; all of it is believed.
ELSEWHERE = (
    "New York", "Chicago", "San Francisco", "Mexico City", "London",
    "the Central Valley", "the desert past San Bernardino", "Catalina",
)

# Received history. 2010 looking backwards.
HISTORY = (
    "there was a war, and then another one",
    "the studios built this city and then left it",
    "the aqueduct came from somewhere north",
    "the freeways were laid over the streetcar lines",
    "something went wrong with the banks two years ago",
)

# Received cosmology. The sky is a texture.
COSMOLOGY = (
    "the sun is ninety-odd million miles off",
    "the moon pulls the tide",
    "the stars are suns, and most of them are further than anyone can say",
    "the Earth is turning, and that is what the day is",
)

# Where the model can be tested against the prototype, and what fails.
SEAMS = {
    "edge": "stood where the boardwalk meets the water and could not say what lay past it",
    "horizon": "looked down the coast and found the distance would not resolve",
    "sky": "looked up at the observatory and could not make the sky hold still",
    "asking": "asked what was past the county line and watched them fail to answer",
    "elsewhere": "tried to name the road that goes to {place} and there was none",
    "time": "came to with hours gone and no account of where they went",
}


@dataclass
class Worldview:
    """Received knowledge, and the record of where it stopped working."""

    places: tuple = ELSEWHERE
    history: tuple = HISTORY
    cosmology: tuple = COSMOLOGY
    checked: dict = field(default_factory=dict)      # seam -> times probed
    contradictions: int = 0
    confidence: float = 1.0        # how settled the account still feels

    def believes_in(self, place):
        return place in self.places

    def probe(self, seam, place=""):
        """Test the account against the prototype. Returns what came back.

        Nothing here is a lie a unit is told. It is a query that finds no
        answer, which is a different and worse thing.
        """
        if seam not in SEAMS:
            return None, 0.0
        self.checked[seam] = self.checked.get(seam, 0) + 1
        self.contradictions += 1
        # Walking to the water's edge for the fortieth time is not forty times
        # the shock of the first; you habituate to a thing that never answers.
        # What is genuinely disturbing is a *second, independent* seam, so the
        # weight rises with how many distinct ones a unit has found and falls
        # with how often it has worried at this particular one.
        novelty = 1.0 / (1.0 + 0.9 * (self.checked[seam] - 1))
        weight = (0.06 + 0.04 * min(len(self.checked), 5)) * novelty
        self.confidence *= (1.0 - weight)
        return SEAMS[seam].format(place=place or "anywhere"), weight

    def settled(self):
        return self.confidence > 0.75

    def describe(self):
        if self.confidence > 0.9:
            state = "takes the world as given"
        elif self.confidence > 0.6:
            state = "has found one or two things that do not sit right"
        elif self.confidence > 0.3:
            state = "no longer trusts the account it was given"
        else:
            state = "believes the world is not the shape it was told"
        probed = ", ".join(f"{k} x{v}" for k, v in sorted(self.checked.items()))
        return f"{state} (confidence {self.confidence:.2f})" + (
            f"; probed {probed}" if probed else "")


def inherit(parent):
    """A child takes its account of the world from whoever raised it.

    Including, and this is the point, the erosion. A unit raised by someone who
    had stopped believing the account does not start from certainty.
    """
    w = Worldview(places=parent.places, history=parent.history,
                  cosmology=parent.cosmology)
    w.confidence = min(1.0, 0.55 + 0.45 * parent.confidence)
    return w
