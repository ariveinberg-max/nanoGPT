"""What a unit can actually find out, as opposed to what is true.

Until now every unit read the world directly. `world.venues_of("food")`
meant a newborn knew every restaurant in Los Angeles, its hours and its
prices. `_crowd_by_venue()` meant it knew exactly who was in a room before
entering it. `_worth_robbing` read `other.funds`, so a unit knew to the cent
how much cash every person nearby was carrying.

That is not a missing feature, it is the one thing that has to be false. An
agent with access to the true state of the world is not embedded in it; it is
observing it from outside, which is the opposite of having a life in it. So
nothing in the decision loop touches the simulation any more. A unit gets a
`View`: what is in front of it right now, thinned by attention, plus what it
personally remembers or was told, with everything else simply absent.

The important consequence is that a unit can be wrong -- about a price, about
whether a place is open, about who is dangerous, about how much someone has in
their pocket -- and act on it, and find out.
"""

from dataclasses import dataclass, field

from . import world


@dataclass
class Known:
    """What a unit believes about a venue. Not what is true about it."""

    key: str
    district: str
    kind: str
    name: str
    cost: float = 0.0           # believed price, which may be out of date
    wage: float = 0.0
    opens: int = 0
    closes: int = 0
    confidence: float = 0.35    # how sure, from how often it has been here
    visits: int = 0

    def open_at(self, hour):
        """A belief about opening hours, and beliefs can be wrong."""
        if self.opens == self.closes:
            return True
        if self.opens < self.closes:
            return self.opens <= hour < self.closes
        return hour >= self.opens or hour < self.closes


@dataclass
class Sighting:
    """Another unit, as seen. Funds are estimated from appearance, not read."""

    name: str
    district: str
    apparent_means: float       # a guess, and a poor one
    familiar: float             # how well this unit knows them
    looks_rough: float = 0.0    # visible signs of a bad time of it


@dataclass
class View:
    """One unit's slice of the world for one decision."""

    here: object = None                      # the venue it is standing in
    present: tuple = ()                      # Sightings, here and now
    places: dict = field(default_factory=dict)   # key -> Known
    hour: int = 0
    is_workday: bool = True

    def of_kind(self, kind):
        return [p for p in self.places.values() if p.kind == kind]

    def sighting(self, name):
        for s in self.present:
            if s.name == name:
                return s
        return None


def learn_venue(u, venue, rng, exact=False):
    """Record a place a unit has actually been.

    Prices and hours are taken down imperfectly the first time and sharpen
    with repetition, which is why a unit will occasionally cross town to a
    place it believes is open and find it shut.
    """
    known = u.known_places.get(venue.key)
    if known is None:
        slip = 0.0 if exact else float(rng.normal(0, 0.18))
        hour_slip = 0 if exact else int(rng.integers(-1, 2))
        known = u.known_places[venue.key] = Known(
            key=venue.key, district=venue.district, kind=venue.kind,
            name=venue.name,
            cost=max(0.0, venue.cost * (1.0 + slip)),
            wage=venue.wage,
            opens=(venue.opens + hour_slip) % 24 if venue.opens != venue.closes
            else venue.opens,
            closes=venue.closes,
            confidence=0.35 if not exact else 0.9,
        )
    known.visits += 1
    # being here corrects the record
    known.confidence = min(1.0, known.confidence + 0.15)
    known.cost += 0.5 * (venue.cost - known.cost)
    known.opens, known.closes = venue.opens, venue.closes
    return known


def hear_of_venue(u, venue, credibility):
    """Told about somewhere, rather than having been. Vaguer, and revisable."""
    if venue.key in u.known_places:
        return
    u.known_places[venue.key] = Known(
        key=venue.key, district=venue.district, kind=venue.kind,
        name=venue.name, cost=venue.cost * 1.1, wage=venue.wage,
        opens=venue.opens, closes=venue.closes,
        confidence=0.2 * credibility)


def estimate_means(observer, other, rng):
    """How well off somebody looks. A guess, biased by what you already think."""
    truth = other.funds
    noise = float(rng.normal(0, 0.45))
    rel = observer.social.people.get(other.name)
    known = rel.familiarity if rel else 0.0
    # you can read someone you know much better than a stranger
    spread = 0.55 * (1.0 - known)
    return max(0.0, truth * (1.0 + noise * spread))


def perceive(sim, u):
    """The projection this unit acts on. Nothing here reads world state."""
    clock = sim.clock
    here = None
    if u.location_key:
        v = world.VENUES_BY_KEY.get(u.location_key)
        if v is not None:
            here = learn_venue(u, v, u.rng)

    present = []
    if u.location_key:
        for other in sim._crowd_by_venue().get(u.location_key, ()):
            if other is u:
                continue
            rel = u.social.people.get(other.name)
            known = rel.familiarity if rel else 0.0
            # visible hardship: thin, unwell, badly turned out. Noisy, and
            # easier to read on somebody you know.
            rough = max(0.0, min(1.0,
                0.5 * other.hunger + 0.5 * (1.0 - other.health) + 0.4 * other.pain
                + float(u.rng.normal(0, 0.28 * (1.0 - known)))))
            sighting = Sighting(
                name=other.name, district=other.district,
                apparent_means=estimate_means(u, other, u.rng),
                familiar=known, looks_rough=rough)
            present.append(sighting)
            # seeing somebody updates what you think is going on with them
            u.social.of(other.name).mind.saw(hardship=rough)

    return View(here=here, present=tuple(present), places=u.known_places,
                hour=clock.hour, is_workday=clock.is_workday)


def notice_surroundings(u, district):
    """You do not have to go in to know a place is there."""
    for v in world.venues_in(district):
        hear_of_venue(u, v, credibility=0.9)


def swap_knowledge(a, b, rng, n=2):
    """Word of mouth. How anyone finds out about anywhere they have not been.

    Knowledge of the city spreads the same way fear of it does: through people
    you have reason to believe.
    """
    for teller, listener in ((a, b), (b, a)):
        rel = listener.social.people.get(teller.name)
        credibility = 0.35 + 0.65 * (rel.trust if rel else 0.5)
        unknown = [k for k in teller.known_places if k not in listener.known_places]
        if not unknown:
            continue
        for key in rng.choice(unknown, size=min(n, len(unknown)), replace=False):
            v = world.VENUES_BY_KEY.get(str(key))
            if v is not None:
                hear_of_venue(listener, v, credibility)


def seed_local_knowledge(u, rng):
    """What a unit starts out knowing: home, work, and its own neighbourhood.

    Nobody is born knowing the city. A unit knows where it lives, where it
    works if it works, and roughly what is on its own streets; everything
    else has to be walked into or heard about.
    """
    for key in (u.home_key, u.work_key):
        v = world.VENUES_BY_KEY.get(key)
        if v is not None:
            learn_venue(u, v, rng, exact=True)
    for v in world.venues_in(u.home.district):
        hear_of_venue(u, v, credibility=1.0)
