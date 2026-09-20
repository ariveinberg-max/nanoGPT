"""Deliberation: how a unit decides, and how what happens lands on it.

The previous build chose an abstract intent from a softmax and looked no
further. Here a unit builds concrete options -- eat *at Philippe's*, go out
*to the Echo* hoping *Ruth Okada* is there -- and scores each against
everything it knows: what it needs, what it can afford, what it believes about
that district, who it expects to find, what it thinks it is any good at, and
what it is feeling. Every option carries the reasons it scored as it did, so a
unit's choice can be read back rather than guessed at.

Learning is kept where it pays. A small policy network supplies a habit prior
over intent categories, trained on realised reward as before; the self-model
learns what to expect; memory learns which districts hurt; relationships learn
who to rely on. What is hand-built is the structure those things plug into,
because appraisal, recall and self-concept have shape that reinforcement does
not discover in a few thousand decisions.
"""

from dataclasses import dataclass, field

import numpy as np

from . import lifecourse, world
from .affect import NEGATIVE, Appraisal
from .brain import INTENTS
from .memory import Episode

# How an event is appraised. Each entry is a template; the code below fills in
# magnitudes from the situation.
EFFORTFUL = ("work", "errand", "socialize", "seek", "wander")

# Roughly how long a unit expects to stay once it gets there. The fare is paid
# against the whole visit, not against the first hour of it: charging a full
# commute to one hour made a shift across town score exactly zero, and units
# with a long ride to work simply stopped going.
STAY_HOURS = {"work": 4.0, "sleep": 5.0, "socialize": 2.0, "eat": 1.0}


@dataclass
class Option:
    intent: str
    venue: str = ""              # where it is done; "" means here
    person: str = ""             # who the unit hopes to find
    score: float = 0.0
    reasons: dict = field(default_factory=dict)

    def label(self, sim=None):
        v = world.VENUES_BY_KEY.get(self.venue)
        where = f" at {v.name}" if v else ""
        who = f" (hoping for {self.person})" if self.person else ""
        return f"{self.intent}{where}{who}"

    def why(self, n=3):
        rows = sorted(self.reasons.items(), key=lambda kv: -abs(kv[1]))[:n]
        return ", ".join(f"{k} {v:+.2f}" for k, v in rows)


# ---------------------------------------------------------------- generation
def generate(sim, u):
    """The options this unit can actually see from where it stands."""
    clock = sim.clock
    opts = [Option("rest"), Option("sleep", u.home_key)]

    if (u.workplace.open_at(clock.hour) and clock.is_workday and u.employed
            and lifecourse.can_work(u.age)):
        opts.append(Option("work", u.work_key))

    for v in world.venues_of("food"):
        if v.open_at(clock.hour) and v.cost <= u.funds:
            opts.append(Option("eat", v.key))
    if u.funds >= world.HOME_MEAL_COST:
        opts.append(Option("eat", u.home_key))

    crowd = sim.district_population()
    for v in world.venues_of("social"):
        if v.open_at(clock.hour) and v.cost <= u.funds:
            hoped = _who_might_be_there(u, crowd.get(v.district, ()))
            opts.append(Option("socialize", v.key, hoped))
    for v in world.venues_of("civic"):
        if v.open_at(clock.hour):
            opts.append(Option("socialize", v.key))

    if u.funds >= world.ERRAND_COST:
        opts.append(Option("errand"))
    opts.append(Option("wander"))
    opts.append(Option("seek"))
    return opts


def _who_might_be_there(u, present):
    """The person this unit is hoping to run into, if it is hoping at all."""
    best, weight = "", 0.0
    for other in present:
        if other is u:
            continue
        r = u.social.people.get(other.name)
        if r and r.closeness() > weight:
            best, weight = other.name, r.closeness()
    return best if weight > 0.15 else ""


# ---------------------------------------------------------------- evaluation
def evaluate(sim, u, opt):
    """Score one option, recording why."""
    clock = sim.clock
    v = world.VENUES_BY_KEY.get(opt.venue)
    r = {}

    # --- what it would do for me ------------------------------------------
    peril = u.peril()
    if opt.intent == "eat":
        relief = 0.85 if v and v.kind == "food" else 0.55
        r["hunger"] = 2.2 * u.hunger * relief
        if peril > 0.1:
            r["survival"] = 4.0 * peril * relief
    elif opt.intent == "sleep":
        from .unit import circadian
        r["fatigue"] = 2.0 * u.fatigue * circadian(clock)
    elif opt.intent == "rest":
        r["fatigue"] = 0.7 * u.fatigue
    elif opt.intent == "work":
        # Wanting money does not switch off the moment there is some. With a
        # three-day horizon and no floor, any unit with a small buffer stopped
        # going to work entirely, then lost the job a fortnight later.
        need = 0.25 + 0.75 * (1.0 - min(1.0, u.funds / (8 * world.DAILY_COST)))
        r["money"] = 1.6 * need * (u.workplace.wage / world.DAILY_COST)
        r["competence"] = 0.5 * (u.selfmodel.can("work") - 0.5)
        r["duty"] = 0.4 * u.selfmodel.roles["worker"]
        if peril > 0.1 and u.funds < 2 * world.HOME_MEAL_COST:
            r["survival"] = 3.0 * peril      # no money and no time left to lose
        if u.debt > 0:
            r["debt"] = 0.6 * min(1.0, u.debt / (5 * world.DAILY_COST))
    elif opt.intent == "socialize":
        r["loneliness"] = 1.5 * u.loneliness
        if opt.person:
            rel = u.social.of(opt.person)
            r["company"] = 1.2 * rel.closeness()
        r["social_self"] = 0.4 * (u.selfmodel.can("social") - 0.5)
    elif opt.intent == "errand":
        r["errand"] = 0.25 * u.hunger
    elif opt.intent == "seek":
        r["restlessness"] = 0.4 * u.traits["restlessness"]

    # --- what it would cost me --------------------------------------------
    if v is not None:
        trip = world.travel_ticks(u.district, v.district)
        r["travel"] = -0.14 * trip / STAY_HOURS.get(opt.intent, 1.0)
        if v.cost:
            r["cost"] = -1.1 * v.cost / world.DAILY_COST
            if u.funds < 2 * world.DAILY_COST:
                r["cost"] *= 2.0          # scarcity makes every dollar louder
    elif opt.intent == "errand":
        r["cost"] = -1.1 * world.ERRAND_COST / world.DAILY_COST

    # --- what I believe about out there -----------------------------------
    district = v.district if v is not None else u.district
    dread = u.memory.dread(clock.tick, district=district,
                           place=opt.venue or u.location_key)
    danger = u.memory.danger_of(district)
    if dread or danger:
        courage = 0.35 + 0.65 * u.selfmodel.can("coping")
        r["fear"] = -(1.4 * dread + 0.9 * danger) / courage
    if opt.person:
        r["wariness"] = -1.0 * u.social.of(opt.person).wariness()

    # --- what I am feeling -------------------------------------------------
    if opt.intent in EFFORTFUL:
        r["drive"] = (u.affect.drive() - 1.0) * 1.2
    if u.pain > 0.1 and opt.intent in EFFORTFUL:
        r["pain"] = -1.3 * u.pain
    if u.affect.intensity["fear"] > 0.2 and opt.intent == "sleep":
        r["retreat"] = 0.8 * u.affect.intensity["fear"]
    if u.affect.intensity["shame"] > 0.2 and opt.intent == "socialize":
        r["shame"] = -1.1 * u.affect.intensity["shame"]
    if u.affect.intensity["anger"] > 0.3 and opt.intent == "seek":
        r["anger"] = 0.6 * u.affect.intensity["anger"]

    # --- habit, and what I have come to expect -----------------------------
    r["habit"] = 0.8 * float(u.habit_prior[INTENTS.index(opt.intent)])
    r["expected"] = 0.5 * u.selfmodel.expect(_key(opt))

    opt.reasons = r
    opt.score = sum(r.values())
    return opt.score


def _key(opt):
    return f"{opt.intent}:{opt.venue}" if opt.venue else opt.intent


def dwell(u):
    """What a unit feels about its situation, as opposed to about an event.

    Hunger, debt, pain and being out of work are conditions, not things that
    happen, so nothing appraised them and a unit could starve without ever
    becoming distressed. They are appraised here every hour, weakly, which is
    what turns a bad month into chronic stress rather than a series of
    unrelated bad hours.
    """
    load = []
    if u.hunger > 0.55:
        load.append(Appraisal(valence=-1.5 * (u.hunger - 0.55), anticipated=True,
                              certainty=0.7, control=0.3))
    if u.debt > 3 * world.DAILY_COST:
        load.append(Appraisal(valence=-min(0.7, u.debt / (14 * world.DAILY_COST)),
                              anticipated=True, certainty=0.6, control=0.25))
    if u.pain > 0.15:
        load.append(Appraisal(valence=-u.pain, control=0.2, irreversible=0.3))
    if not u.employed:
        load.append(Appraisal(valence=-0.5, anticipated=True, certainty=0.5,
                              control=0.3))
    if u.peril() > 0.15:
        # Fear of dying, which is the only reason any of the rest of it counts.
        load.append(Appraisal(valence=-1.4 * u.peril(), anticipated=True,
                              certainty=0.55, control=0.35 * u.selfmodel.can("coping"),
                              irreversible=1.0))
    if u.loneliness > 0.7:
        load.append(Appraisal(valence=-0.4 * u.traits["sociability"],
                              control=0.5, irreversible=0.2))
    # Hourly, against per-tick decay: a condition has to be re-felt about
    # four times just to hold its ground.
    for a in load:
        u.affect.feel(a.emotions(), scale=0.45)


def deliberate(sim, u):
    """Intrusive recall, then options, then a choice."""
    ep = u.memory.intrude(sim.clock.tick, u.rng, u.affect.trauma)
    if ep is not None:
        u.affect.feel(ep.emotions, scale=0.35)
        u.rumination += 1
    dwell(u)

    u.habit_prior = u.policy.forward(u.observe(sim.clock))[1]
    opts = generate(sim, u)
    for o in opts:
        evaluate(sim, u, o)

    # Bounded consideration. A unit does not weigh thirteen bars against one
    # bed: it brings to mind the best option of each kind and chooses among
    # those. Without this, an intent with many venues won the softmax on sheer
    # count -- units socialised 38% of the time while starving, because there
    # were thirteen ways to socialise and one way to sleep.
    best = {}
    for o in opts:
        if o.intent not in best or o.score > best[o.intent].score:
            best[o.intent] = o
    opts = list(best.values())

    scores = np.array([o.score for o in opts])
    # Stress narrows the field. Under load a unit stops weighing and reaches
    # for whatever it reaches for, which is how bad situations entrench.
    temp = max(0.25, 0.9 - 0.5 * u.affect.stress)
    z = (scores - scores.max()) / temp
    p = np.exp(z)
    p /= p.sum()
    chosen = opts[int(u.rng.choice(len(opts), p=p))]
    u.considered = opts
    u.chose = chosen
    return chosen


def flail(sim, u):
    """The control: the same options, chosen without weighing any of them."""
    dwell(u)
    u.habit_prior = u.policy.forward(u.observe(sim.clock))[1]
    opts = generate(sim, u)
    best = {}
    for o in opts:
        if o.intent not in best or u.rng.random() < 0.5:
            best[o.intent] = o
    opts = list(best.values())
    chosen = opts[int(u.rng.integers(len(opts)))]
    u.considered = opts
    u.chose = chosen
    return chosen


# ---------------------------------------------------------------- experience
def probe_world(sim, u, seam, place=""):
    """A unit tests its account of the world against the prototype.

    What comes back is not a lie it is told. It is a query that finds no
    answer, which is a different and worse thing, and it is appraised the way
    anything unresolved and uncontrollable is appraised: as fear.
    """
    text, weight = u.worldview.probe(seam, place)
    if text is None:
        return
    experience(sim, u, f"seam_{seam}", text,
               Appraisal(valence=-0.25 - 2.5 * weight, anticipated=True,
                         certainty=0.35, control=0.05, irreversible=0.5))


def experience(sim, u, kind, text, appraisal, place="", people=()):
    """An event happens to a unit: appraise it, feel it, remember it."""
    emotions = appraisal.emotions()
    novelty = 1.0 if not any(e.kind == kind for e in u.memory.episodes[-30:]) else 0.0
    u.affect.feel(emotions)
    # The worst single events leave something behind even when the summed
    # intensity of the moment does not clear the chronic-load threshold: four
    # bereavements at maximum salience were leaving a unit with no trauma.
    weight = sum(emotions.get(e, 0.0) for e in NEGATIVE)
    if weight > 0.8:
        u.affect.trauma = min(1.0, u.affect.trauma + 0.06 * (weight - 0.8))
    district = (world.VENUES_BY_KEY[place].district if place in world.VENUES_BY_KEY
                else u.district)
    u.memory.encode(Episode(tick=sim.clock.tick, kind=kind, text=text,
                            place=place, district=district, people=tuple(people),
                            emotions=emotions), novelty=novelty)
    return emotions
