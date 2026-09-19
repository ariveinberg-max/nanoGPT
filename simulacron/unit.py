"""A unit: a fully formed, self-learning cyber being.

Body, feeling, memory, self-concept and standing with other people, all of it
running whether or not anyone is watching. What the decision loop reads is in
`observe` and in the systems hung off this object; what a unit *is*, as far as
the simulation is concerned, is the whole of it.
"""

from dataclasses import dataclass, field

import numpy as np

from . import world
from .affect import Affect
from .brain import INTENTS, N_INTENTS, Learner, Policy
from .memory import Memory
from .selfmodel import SelfModel
from .social import Social

FIRST_NAMES = ("Ada", "Vernon", "Ruth", "Cleo", "Hollis", "Marguerite", "Sol",
               "Ines", "Lucien", "Dorothea", "Amos", "Bess", "Rafael", "Ottoline",
               "Gus", "Selma", "Emmett", "Nadia", "Tobias", "June")
LAST_NAMES = ("Ashgrove", "Kestrel", "Dunbar", "Moreno", "Vandevere", "Okada",
              "Hale", "Bramwell", "Ferris", "Quintana", "Lindqvist", "Ives",
              "Castellane", "Rourke", "Pym", "Abernathy")

# The habit layer's view. Deliberation sees far more than this -- memories,
# relationships, options with reasons attached -- but the learned prior over
# intent categories runs on a fixed-width vector, and this is it.
OBS_FIELDS = (
    "hunger", "fatigue", "loneliness", "pain", "funds", "debt",
    "valence", "arousal", "stress", "trauma", "drive",
    "esteem", "prospects", "worker_role",
    "hour_sin", "hour_cos", "is_workday", "workplace_open", "employed",
    "at_home", "at_work", "here_danger", "dissonance",
    "sociability", "diligence", "appetite", "restlessness", "thrift",
) + tuple(f"in_{d}" for d in world.DISTRICTS)
N_OBS = len(OBS_FIELDS)


def pick_job(rng, home_district):
    """A job someone living here would plausibly take.

    Drawn at random it handed people a daily round trip across the county, and
    they rationally stopped going. Weighting by how far it is from home is
    what most people actually do about that.
    """
    jobs = world.venues_of("work")
    w = np.array([1.0 / (1.0 + world.travel_ticks(home_district, v.district))
                  for v in jobs])
    return jobs[int(rng.choice(len(jobs), p=w / w.sum()))]


def circadian(clock):
    """Sleep pressure over the day: highest around 3am, lowest mid-afternoon.

    Without this a unit slept as happily at two in the afternoon as at two in
    the morning, and the population never settled into nights. It is the
    plainest fact about being modeled after us.
    """
    hour = clock.hour + clock.minute / 60.0
    return 1.0 + 0.55 * np.cos(2 * np.pi * (hour - 3.0) / 24.0)


@dataclass
class Unit:
    name: str
    home_key: str
    work_key: str
    district: str
    traits: dict
    rng: object

    # -- body ---------------------------------------------------------------
    hunger: float = 0.25
    fatigue: float = 0.25
    loneliness: float = 0.25
    pain: float = 0.0
    health: float = 1.0
    funds: float = 0.0
    debt: float = 0.0
    employed: bool = True

    # -- mind ---------------------------------------------------------------
    affect: Affect = field(default_factory=Affect)
    memory: Memory = field(default_factory=Memory)
    social: Social = field(default_factory=Social)
    selfmodel: SelfModel = field(default_factory=SelfModel)
    rumination: int = 0
    dissonance: float = 0.0

    # -- where it is and what it is doing -----------------------------------
    location_key: str = ""
    travel_left: int = 0
    travel_to: str = ""
    intent: str = "rest"
    activity: str = "idle"
    target: str = ""
    linked: bool = False
    alive_ticks: int = 0

    # -- machinery ----------------------------------------------------------
    worked_last: bool = False
    earned: float = 0.0
    hold_left: int = 0
    pending_record: object = None
    pending_reward: float = 0.0
    habit_prior: object = None
    considered: list = field(default_factory=list)
    chose: object = None
    policy: object = None
    learner: object = None

    # -- construction -------------------------------------------------------
    @classmethod
    def spawn(cls, rng):
        home = rng.choice([v for v in world.VENUES if v.kind == "home"])
        work = pick_job(rng, home.district)
        traits = {
            "sociability": float(rng.uniform(0.2, 1.0)),
            "diligence": float(rng.uniform(0.2, 1.0)),
            "appetite": float(rng.uniform(0.6, 1.4)),
            "restlessness": float(rng.uniform(0.1, 0.9)),
            "thrift": float(rng.uniform(0.2, 1.0)),
        }
        u = cls(
            name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            home_key=home.key, work_key=work.key, district=home.district,
            traits=traits, rng=rng, location_key=home.key,
            funds=world.STARTING_FUNDS,
        )
        u.selfmodel.efficacy["work"] = 0.35 + 0.4 * traits["diligence"]
        u.selfmodel.efficacy["social"] = 0.35 + 0.4 * traits["sociability"]
        u.selfmodel.roles["worker"] = 0.3 * traits["diligence"]
        u.policy = Policy(N_OBS, rng=rng)
        u.learner = Learner(u.policy)
        u.habit_prior = np.full(N_INTENTS, 1.0 / N_INTENTS)
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
        return 1.0 - (self.hunger + self.fatigue + self.loneliness + self.pain) / 4.0

    # -- perception ---------------------------------------------------------
    def observe(self, clock):
        hour_frac = (clock.tick % world.TICKS_PER_DAY) / world.TICKS_PER_DAY
        a = self.affect
        obs = [
            self.hunger, self.fatigue, self.loneliness, self.pain,
            min(self.funds / (15 * world.DAILY_COST), 1.5),
            min(self.debt / (10 * world.DAILY_COST), 1.5),
            a.valence(), a.arousal(), a.stress, a.trauma, a.drive(),
            self.selfmodel.esteem(), self.selfmodel.prospects,
            self.selfmodel.roles["worker"],
            np.sin(2 * np.pi * hour_frac), np.cos(2 * np.pi * hour_frac),
            1.0 if clock.is_workday else 0.0,
            1.0 if self.workplace.open_at(clock.hour) else 0.0,
            1.0 if self.employed else 0.0,
            1.0 if self.location_key == self.home_key else 0.0,
            1.0 if self.location_key == self.work_key else 0.0,
            self.memory.danger_of(self.district),
            min(self.dissonance, 1.0),
            self.traits["sociability"], self.traits["diligence"],
            self.traits["appetite"], self.traits["restlessness"],
            self.traits["thrift"],
        ]
        obs += [1.0 if self.district == d else 0.0 for d in world.DISTRICTS]
        return np.array(obs, dtype=float)

    # -- drives -------------------------------------------------------------
    def decay(self, clock):
        self.hunger = min(1.0, self.hunger + 0.011 * self.traits["appetite"])
        self.fatigue = min(1.0, self.fatigue + 0.011 * circadian(clock))
        self.loneliness = min(1.0, self.loneliness + 0.006 * self.traits["sociability"])
        if self.health < 1.0:
            self.health = min(1.0, self.health + 0.0015)
            self.pain = max(0.0, self.pain - 0.004)
        self.affect.decay()
        self.social.forget()
        self.alive_ticks += 1

    def reward(self):
        """What the habit layer is trained on: a unit's own sense of the hour."""
        r = self.wellbeing()
        r += 0.60 * min(self.funds / (2.5 * world.DAILY_COST), 1.0)
        r += 3.0 * self.earned / world.DAILY_COST
        if self.worked_last:
            r += 0.12 * self.traits["diligence"]
        r -= 1.0 * max(0.0, self.hunger - 0.50)
        r -= 0.8 * max(0.0, self.fatigue - 0.50)
        r -= 3.0 * max(0.0, self.fatigue - 0.75)
        r -= 0.5 * max(0.0, self.loneliness - 0.60)
        r += 0.45 * self.affect.valence()
        r -= 0.40 * self.affect.stress
        r -= 0.60 * self.pain
        r -= 0.30 * min(1.0, self.debt / (5 * world.DAILY_COST))
        return r

    def remember(self, tick, text, weight=1.0):
        """Plain note with no emotional content, for the link shell."""
        from .memory import Episode
        self.memory.encode(Episode(tick=tick, text=text, kind="note",
                                   district=self.district,
                                   emotions={"sadness": 0.2 * weight}))

    def recent_memories(self, n=6):
        return self.memory.recent(n)

    # -- readout ------------------------------------------------------------
    def describe(self):
        def bar(x):
            return "#" * int(round(max(0.0, min(1.0, x)) * 10)) + "." * (
                10 - int(round(max(0.0, min(1.0, x)) * 10)))
        where = self.location.name if self.location else "in transit"
        lines = [
            f"{self.name}",
            f"  lives   {self.home.name} ({self.home.district})",
            f"  works   {self.workplace.name} at ${self.workplace.wage:.2f}/hr"
            + ("" if self.employed else "  [OUT OF WORK]"),
            f"  now at  {where} in {self.district}",
            f"  hunger      [{bar(self.hunger)}]   fatigue [{bar(self.fatigue)}]",
            f"  loneliness  [{bar(self.loneliness)}]   pain    [{bar(self.pain)}]",
            f"  funds       ${self.funds:,.2f}"
            + (f"   debt ${self.debt:,.2f}" if self.debt > 0.01 else ""),
            f"  feeling     {self.affect.describe()}",
            f"  self        {self.selfmodel.narrative(self.affect)}",
            f"  dissonance  [{bar(min(self.dissonance, 1.0))}]",
        ]
        if self.social.people:
            lines.append("  knows       " + "; ".join(self.social.describe(2)))
        beliefs = self.memory.describe_places()
        if beliefs:
            lines.append("  believes    " + beliefs[0])
        if self.chose is not None:
            lines.append(f"  chose       {self.chose.label()} -- {self.chose.why()}")
        return "\n".join(lines)
