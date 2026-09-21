"""Crime, policing, and how fear gets around.

Robbery is not a special mode a unit enters. It is an option on the same list
as going to work and going to bed, scored by the same machinery: what it would
get me, what it would cost me, what I believe about being caught, and what I
would have to think of myself afterwards. A unit robs someone when that sum
comes out ahead, which is mostly when it is desperate and thinks nobody is
watching -- and never when the rent is paid and the children are fed.

Policing is deliberately crude and deliberately reflexive: patrol follows
reported crime. That one line is enough to produce the loop everyone already
knows about. More patrol in a district means more arrests there, which means
more recorded crime there, which means more patrol. The distribution of
attention drifts away from where offences happen and towards where offences
are *seen*, and nothing in the code says so.

Fear travels faster than crime does. A robbery is experienced by one person
and heard about by dozens, each of whom adjusts what they believe about a
district they may never have been to, and then stays away from it.
"""

from dataclasses import dataclass, field

from . import world
from .affect import Appraisal

# A 45-day sentence against a flat arrest chance put a hundred of two hundred
# and fifty units inside. Real cities do not hold forty per cent of their
# population in prison; the sentence is shorter, the arrest less certain, and
# most offences never reach anybody official at all.
JAIL_DAYS = 18
BASE_ARREST = 0.09


@dataclass
class Report:
    tick: int
    district: str
    reported: bool


class Police:
    """Attention, and where it goes."""

    def __init__(self, districts=world.DISTRICTS, strength=1.0):
        n = len(districts)
        self.patrol = {d: 1.0 / n for d in districts}
        self.strength = strength
        self.reports = {d: 0.0 for d in districts}
        self.arrests = {d: 0 for d in districts}
        self.offences = {d: 0 for d in districts}

    def record(self, district, reported):
        self.offences[district] = self.offences.get(district, 0) + 1
        if reported:
            self.reports[district] = self.reports.get(district, 0.0) + 1.0

    def reallocate(self):
        """Patrol follows reported crime. This is the whole of the policy."""
        for d in self.reports:
            # Half-life of about four months. At 0.97 a day the record decayed
            # faster than a low crime rate could feed it, total went to nothing
            # between offences, and patrol never left uniform.
            self.reports[d] *= 0.995
        n = len(self.patrol)
        total = sum(self.reports.values())
        if total <= 0:
            return
        # Smoothed against a prior, so a single report in eight districts does
        # not capture most of the city's attention. Unsmoothed, one robbery in
        # Boyle Heights took patrol there to 69% and left it.
        # The prior has to grow with the evidence. Fixed at 2.5 it was swamped
        # once reports ran to the hundreds, every district looked alike, and
        # the feedback loop that produces policing bias got *weaker* at scale
        # -- concentration fell from 1.31x even to 1.1x, which is backwards.
        prior = max(2.5, 0.12 * total)
        for d in self.patrol:
            want = (self.reports[d] + prior) / (total + prior * n)
            target = 0.5 / n + 0.5 * want
            self.patrol[d] += 0.05 * (target - self.patrol[d])

    def presence(self, district):
        return self.strength * self.patrol.get(district, 0.0) * len(self.patrol)

    def arrest_chance(self, district, witnesses):
        p = BASE_ARREST * self.presence(district)
        p += 0.06 * min(witnesses, 4)
        return min(0.9, p)

    def describe(self):
        rows = sorted(self.patrol.items(), key=lambda kv: -kv[1])
        return [f"{d:14s} patrol {w:5.1%}  offences {self.offences.get(d,0):3d}"
                f"  arrests {self.arrests.get(d,0):3d}" for d, w in rows]


def temptation(u, seen, police, sim):
    """What the unit expects to get, and what it expects it to cost.

    Returned as a dict so the reasons show up in the option's own accounting,
    like every other term.
    """
    # What they look like they are carrying, not what they are carrying.
    take = min(seen.apparent_means, 3 * world.DAILY_COST)
    watched = police.presence(u.district)
    # Conscience has a floor. Written as a single product of self-concept
    # terms it decayed quadratically with each offence -- past about half an
    # offender identity it reached zero, and the unit robbed somebody every
    # waking hour for the rest of its life. Some of the cost of robbing a
    # person has nothing to do with how you think of yourself.
    identity = (1.0 - u.selfmodel.roles["offender"]) \
        * (0.35 + 0.65 * u.selfmodel.roles["worker"])
    # temperament: how much somebody minds doing this to a person
    decency = 0.6 + 0.8 * (u.person.scale("agreeableness") if u.person else 0.5)
    r = {
        "take": 1.6 * min(1.0, take / (2 * world.DAILY_COST)),
        "desperation": 2.6 * u.peril() + 2.2 * u.dependents_worry(),
        "risk": -2.2 * watched * (0.4 + 0.6 * (1.0 - u.affect.stress)),
        "conscience": -(1.3 + 2.6 * identity) * decency,
    }
    # A well you have already been down. They have nothing left and they are
    # watching for you.
    recent = sum(1 for e in u.memory.episodes[-40:]
                 if e.kind == "robbery" and seen.name in e.people)
    if recent:
        r["been here"] = -1.1 * min(recent, 4)
    rel = u.social.people.get(seen.name)
    if rel is not None and rel.familiarity > 0.2:
        r["known to me"] = -2.6 * rel.closeness()
    return r


def commit(sim, u, victim, police):
    """One robbery, and everything that falls out of it."""
    district = u.district
    crowd = [o for o in sim.units
             if o.location_key == u.location_key and o not in (u, victim)]
    take = min(victim.funds, 3 * world.DAILY_COST)
    victim.funds -= take
    u.funds += take
    hurt = sim.rng.random() < 0.3
    if hurt:
        victim.pain = min(1.0, victim.pain + 0.4)
        victim.health = max(0.05, victim.health - 0.15)

    from . import cognition
    cognition.experience(
        sim, victim, "robbed",
        f"robbed by {u.name} in {district}" if victim.social.people.get(u.name)
        else f"robbed in {district}",
        Appraisal(valence=-0.85 if hurt else -0.7, agency="other", norm=-1.0,
                  control=0.15, public=0.5 if crowd else 0.0,
                  irreversible=0.5 if hurt else 0.2, other=u.name),
        place=victim.location_key, people=(u.name,))
    victim.selfmodel.endorse("victim", 0.3)
    victim.social.treated(u.name, sim.clock.tick, valence=-0.9, betrayal=0.8)
    victim.social.of(u.name).mind.saw(harm=1.0, help=0.0, warmth=0.0)

    for w in crowd:
        # witnesses update what they think this person is capable of
        w.social.of(u.name).mind.saw(harm=0.85, warmth=0.15)
        cognition.experience(
            sim, w, "witnessed", f"saw someone robbed in {district}",
            Appraisal(valence=-0.45, agency="other", norm=-1.0, control=0.3,
                      other=u.name),
            place=w.location_key, people=(u.name,))

    cognition.experience(
        sim, u, "robbery", f"took ${take:.0f} off {victim.name}",
        Appraisal(valence=0.35, agency="self", norm=-0.9,
                  public=0.6 if crowd else 0.1, harmed_other=0.5 + 0.3 * hurt),
        place=u.location_key, people=(victim.name,))
    u.selfmodel.endorse(
        "offender", 0.12 * (1.0 - u.selfmodel.roles["offender"]))
    u.selfmodel.endorse("worker", -0.02)

    # Whether it reaches the police at all depends on the victim, not the crime.
    # Most of it never reaches anybody official.
    reported = (victim.social.of(u.name).familiarity < 0.5
                and sim.rng.random() < 0.18 + 0.28 * min(1.0, len(crowd) / 3))
    police.record(district, reported)

    caught = reported and sim.rng.random() < police.arrest_chance(district, len(crowd))
    if caught:
        police.arrests[district] = police.arrests.get(district, 0) + 1
        sim.imprison(u, JAIL_DAYS)

    for teller in [victim] + crowd:
        rumour(sim, teller, district, 0.55 if teller is victim else 0.35)
    return take, hurt, caught


def rumour(sim, teller, district, weight):
    """What happened to one person becomes what everybody knows about a place.

    Told to a few people, not to everyone a unit knows, and with less force
    each time the listener has already heard it. At n=24 telling the six
    closest was fine; at n=250 every unit knew ten people, everyone heard
    everything, and the fear map went flat -- spread across districts fell
    from 0.57 to 0.17, which kills the one thing this system is for. A story
    has to be able to stay in one part of town.
    """
    from .memory import PlaceBelief
    told = 0
    # you tell the people you are actually close to, and there are not many
    for rel in teller.social.closest(3):
        other = sim.by_name.get(rel.name)
        if other is None or rel.familiarity < 0.35:
            continue
        if sim.rng.random() > 0.55:
            continue                      # it does not come up
        credibility = 0.35 + 0.65 * rel.trust
        b = other.memory.places.setdefault(district, PlaceBelief())
        # diminishing: the tenth time you hear a place is rough moves you less
        room = max(0.0, 1.0 - b.danger)
        b.danger = min(1.0, b.danger + weight * credibility * 0.5 * room)
        b.fear = min(1.0, b.fear + weight * credibility * 0.35 * room)
        told += 1
    return told
