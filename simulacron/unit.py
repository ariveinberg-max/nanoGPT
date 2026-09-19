"""A unit: a fully formed, self-learning cyber being.

It is not an NPC waiting on a user. It carries drives that decay whether or not
anyone is watching, a policy that it trains on its own experience, and an
episodic memory it can be asked about. Modeled after us, which is to say it
gets hungry, it gets tired, it wants company, and it notices when time goes
missing.
"""

from dataclasses import dataclass, field

import numpy as np

from . import world
from .brain import INTENTS, N_INTENTS, Learner, Policy

FIRST_NAMES = ("Ada", "Vernon", "Ruth", "Cleo", "Hollis", "Marguerite", "Sol",
               "Ines", "Lucien", "Dorothea", "Amos", "Bess", "Rafael", "Ottoline",
               "Gus", "Selma", "Emmett", "Nadia", "Tobias", "June")
LAST_NAMES = ("Ashgrove", "Kestrel", "Dunbar", "Moreno", "Vandevere", "Okada",
              "Hale", "Bramwell", "Ferris", "Quintana", "Lindqvist", "Ives",
              "Castellane", "Rourke", "Pym", "Abernathy")

# Observation layout, kept explicit so the vector stays readable in a debugger.
OBS_FIELDS = (
    "hunger", "fatigue", "loneliness", "funds", "mood",
    "hour_sin", "hour_cos", "is_workday", "workplace_open", "at_home",
    "at_work", "employed", "dissonance",
    "sociability", "diligence", "appetite", "restlessness", "thrift",
) + tuple(f"in_{d}" for d in world.DISTRICTS)
N_OBS = len(OBS_FIELDS)


def circadian(clock):
    """Sleep pressure over the day: highest around 3am, lowest mid-afternoon.

    Without this a unit slept as happily at two in the afternoon as at two in
    the morning, and the population never settled into nights. It is the
    plainest fact about being modeled after us.
    """
    hour = clock.hour + clock.minute / 60.0
    return 1.0 + 0.55 * np.cos(2 * np.pi * (hour - 3.0) / 24.0)


@dataclass
class Memory:
    tick: int
    text: str
    weight: float = 1.0


@dataclass
class Unit:
    name: str
    home_key: str
    work_key: str
    district: str
    traits: dict
    rng: object

    hunger: float = 0.25
    fatigue: float = 0.25
    loneliness: float = 0.25
    funds: float = world.STARTING_FUNDS
    mood: float = 0.5
    dissonance: float = 0.0          # accumulated evidence the world is wrong

    location_key: str = ""
    travel_left: int = 0
    travel_to: str = ""
    intent: str = "rest"
    activity: str = "idle"     # what actually happened, not what was wanted
    linked: bool = False             # an operator is riding this body
    alive_ticks: int = 0
    pending_record: object = None
    worked_last: bool = False
    hold_left: int = 0
    earned: float = 0.0
    pending_reward: float = 0.0

    memories: list = field(default_factory=list)
    policy: object = None
    learner: object = None

    # -- construction -------------------------------------------------------
    @classmethod
    def spawn(cls, rng):
        home = rng.choice([v for v in world.VENUES if v.kind == "home"])
        work = rng.choice([v for v in world.VENUES if v.kind == "work"])
        traits = {
            "sociability": float(rng.uniform(0.2, 1.0)),
            "diligence": float(rng.uniform(0.2, 1.0)),
            "appetite": float(rng.uniform(0.6, 1.4)),
            "restlessness": float(rng.uniform(0.1, 0.9)),
            "thrift": float(rng.uniform(0.2, 1.0)),
        }
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        u = cls(
            name=name,
            home_key=home.key,
            work_key=work.key,
            district=home.district,
            traits=traits,
            rng=rng,
            location_key=home.key,
        )
        u.policy = Policy(N_OBS, rng=rng)
        u.learner = Learner(u.policy)
        return u

    # -- accessors ----------------------------------------------------------
    @property
    def home(self):
        return world.VENUES_BY_KEY[self.home_key]

    @property
    def workplace(self):
        return world.VENUES_BY_KEY[self.work_key]

    @property
    def location(self):
        return world.VENUES_BY_KEY.get(self.location_key)

    @property
    def travelling(self):
        return self.travel_left > 0

    def wellbeing(self):
        return 1.0 - (self.hunger + self.fatigue + self.loneliness) / 3.0

    # -- perception ---------------------------------------------------------
    def observe(self, clock):
        hour_frac = (clock.tick % world.TICKS_PER_DAY) / world.TICKS_PER_DAY
        obs = [
            self.hunger,
            self.fatigue,
            self.loneliness,
            min(self.funds / (15 * world.DAILY_COST), 1.5),
            self.mood,
            np.sin(2 * np.pi * hour_frac),
            np.cos(2 * np.pi * hour_frac),
            1.0 if clock.is_workday else 0.0,
            1.0 if self.workplace.open_at(clock.hour) else 0.0,
            1.0 if self.location_key == self.home_key else 0.0,
            1.0 if self.location_key == self.work_key else 0.0,
            1.0,
            min(self.dissonance, 1.0),
            self.traits["sociability"],
            self.traits["diligence"],
            self.traits["appetite"],
            self.traits["restlessness"],
            self.traits["thrift"],
        ]
        obs += [1.0 if self.district == d else 0.0 for d in world.DISTRICTS]
        return np.array(obs, dtype=float)

    # -- drives -------------------------------------------------------------
    def decay(self, clock):
        self.hunger = min(1.0, self.hunger + 0.011 * self.traits["appetite"])
        self.fatigue = min(1.0, self.fatigue + 0.011 * circadian(clock))
        self.loneliness = min(1.0, self.loneliness + 0.006 * self.traits["sociability"])
        self.mood += 0.02 * (self.wellbeing() - self.mood)
        self.alive_ticks += 1

    def reward(self):
        """How well the last tick served this unit, in its own terms.

        Deliberately dense and deliberately linear above the thresholds. A unit
        that only discovered its mistake once it was starving in the street
        would never find its way to a job, and squared penalties made units
        that did nothing all day but manage their own fatigue.
        """
        r = self.wellbeing()
        # Money is felt relative to what a day costs in this era, so the same
        # learner works whether a meal is 35 cents or seven dollars.
        r += 0.60 * min(self.funds / (2.5 * world.DAILY_COST), 1.0)
        r += 3.0 * self.earned / world.DAILY_COST   # the wage itself
        if self.worked_last:
            r += 0.12 * self.traits["diligence"]    # some of them like the work
        r -= 1.0 * max(0.0, self.hunger - 0.50)
        # Being tired is a nuisance; being wrecked is not. The steep second
        # term is what finally sent the population home at night -- with a
        # single linear slope they parked at four fifths exhausted and idled
        # through the small hours rather than going to bed.
        r -= 0.8 * max(0.0, self.fatigue - 0.50)
        r -= 3.0 * max(0.0, self.fatigue - 0.75)
        r -= 0.5 * max(0.0, self.loneliness - 0.60)
        return r

    def remember(self, tick, text, weight=1.0):
        if self.memories and self.memories[-1].text == text:
            return                      # one encounter, not four quarter-hours
        self.memories.append(Memory(tick, text, weight))
        if len(self.memories) > 240:
            del self.memories[:80]

    def recent_memories(self, n=6):
        return self.memories[-n:]

    def describe(self):
        def bar(x):
            filled = int(round(x * 10))
            return "#" * filled + "." * (10 - filled)
        return (
            f"{self.name}\n"
            f"  lives   {self.home.name} ({self.home.district})\n"
            f"  works   {self.workplace.name} at ${self.workplace.wage:.2f}/hr\n"
            f"  now at  {self.location.name if self.location else 'in transit'}"
            f" in {self.district}\n"
            f"  hunger      [{bar(self.hunger)}]\n"
            f"  fatigue     [{bar(self.fatigue)}]\n"
            f"  loneliness  [{bar(self.loneliness)}]\n"
            f"  funds       ${self.funds:,.2f}\n"
            f"  dissonance  [{bar(min(self.dissonance,1.0))}]"
        )
