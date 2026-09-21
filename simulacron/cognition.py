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

from . import crime, dualprocess, goals as goals_mod, lifecourse, perception, workspace, world
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
    """The options this unit can actually see from where it stands.

    A child's world is small, and gets bigger. Left with the adult option set
    a one-year-old was riding across the county on its own and standing at the
    edge of town questioning the nature of reality, which is a memorable image
    and completely wrong.
    """
    view = u.view = u.view or perception.perceive(sim, u)
    child = u.age < lifecourse.ADULT
    near = u.home.district
    opts = [Option("rest"), Option("sleep", u.home_key)]
    if child and u.age < 6.0:
        # too small to go anywhere or do anything about it
        return opts + [Option("eat", u.home_key)]

    job = view.places.get(u.work_key)
    if (job is not None and job.open_at(view.hour) and view.is_workday
            and u.employed and lifecourse.can_work(u.age)):
        opts.append(Option("work", u.work_key))

    for v in view.of_kind("food"):
        if child and v.district != near:
            continue
        if v.open_at(view.hour) and v.cost <= u.funds:
            opts.append(Option("eat", v.key))
    if u.funds >= world.HOME_MEAL_COST:
        opts.append(Option("eat", u.home_key))

    for v in view.of_kind("social"):
        if child and v.district != near:
            continue
        if v.open_at(view.hour) and v.cost <= u.funds:
            opts.append(Option("socialize", v.key, _who_might_be_there(u, v)))
    for v in view.of_kind("civic"):
        if child and v.district != near:
            continue
        if v.open_at(view.hour):
            opts.append(Option("socialize", v.key))

    # Robbery is on the same list as everything else, and is scored by the
    # same machinery. A unit does not enter a criminal mode; it weighs this
    # against going to bed, and mostly goes to bed.
    if not child and u.age < 65:
        mark = _worth_robbing(u)
        if mark is not None:
            opts.append(Option("rob", person=mark.name))

    if u.funds >= world.ERRAND_COST and not child:
        opts.append(Option("errand"))
    opts.append(Option("wander"))
    if not child:
        opts.append(Option("seek"))
    return opts


def _worth_robbing(u):
    """Somebody here who looks worth it. Looks, not is.

    A unit does not case everybody in the room -- it notices one or two. Which
    also stops the best target in a crowd of eighty being found every time,
    and with it the arms race between population density and crime.
    """
    seen = u.view.present
    if not seen:
        return None
    noticed = seen if len(seen) <= 3 else [
        seen[int(u.rng.integers(len(seen)))] for _ in range(3)]
    best = max(noticed, key=lambda s: s.apparent_means)
    return best if best.apparent_means > 0.5 * world.DAILY_COST else None


def _who_might_be_there(u, place):
    """Who a unit expects to find somewhere, from having found them before.

    An expectation built out of its own history, not a reading of where
    everyone currently is.
    """
    best, weight = "", 0.0
    for name, rel in u.social.people.items():
        if rel.closeness() <= weight:
            continue
        seen_here = sum(1 for e in u.memory.episodes[-60:]
                        if e.place == place.key and name in e.people)
        if seen_here:
            best, weight = name, rel.closeness()
    return best if weight > 0.15 else ""


# ---------------------------------------------------------------- evaluation
def evaluate(sim, u, opt):
    """Score one option, recording why."""
    clock = sim.clock
    v = u.view.places.get(opt.venue)
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
        r["money"] = 1.6 * need * (v.wage / world.DAILY_COST if v else 1.0)
        r["competence"] = 0.5 * (u.selfmodel.can("work") - 0.5)
        r["duty"] = 0.4 * u.selfmodel.roles["worker"]
        if peril > 0.1 and u.funds < 2 * world.HOME_MEAL_COST:
            r["survival"] = 3.0 * peril      # no money and no time left to lose
        need = u.dependents_worry(sim.clock.tick)
        if need > 0.05:
            r["dependents"] = 3.2 * need     # somebody at home is going hungry
        if u.debt > 0:
            r["debt"] = 0.6 * min(1.0, u.debt / (5 * world.DAILY_COST))
    elif opt.intent == "socialize":
        r["loneliness"] = 1.5 * u.loneliness
        worry = u.dependents_worry(sim.clock.tick)
        if worry > 0.35:
            r["dependents"] = -1.6 * worry
        if opt.person:
            rel = u.social.of(opt.person)
            r["company"] = 1.2 * rel.closeness()
            # what it expects of them, which is a belief and can be wrong
            r["expects of them"] = 0.7 * rel.mind.expects_help() \
                - 1.1 * rel.mind.expects_harm()
        r["social_self"] = 0.4 * (u.selfmodel.can("social") - 0.5)
    elif opt.intent == "errand":
        r["errand"] = 0.25 * u.hunger
    elif opt.intent == "seek":
        r["restlessness"] = 0.4 * u.traits["restlessness"]
    elif opt.intent == "rob":
        seen = u.view.sighting(opt.person)
        if seen is None:
            r["gone"] = -99.0
        else:
            r.update(crime.temptation(u, seen, sim.police, sim,
                                      present=len(u.view.present)))

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
        rel = u.social.of(opt.person)
        r["wariness"] = -1.0 * rel.wariness()
        if opt.intent == "rob":
            # will they fight back, and will they go to the police
            r["they might"] = -1.3 * rel.mind.expects_harm()

    # --- what I like, and what I care about --------------------------------
    # This is the part no drive can produce. A unit goes to the ocean because
    # it likes the ocean, while not hungry, not lonely and not due anywhere.
    if opt.venue and opt.venue in u.drawn_to:
        # Liking something is a leisure pull, not a drive. Left at full weight
        # it competed with eating, and units went to the ocean hungry.
        spare = max(0.08, 1.0 - max(u.hunger, u.fatigue, 1.5 * u.peril()))
        r["drawn to it"] = (1.15 + 0.5 * u.person.scale("openness")) * spare

    who = u.person
    if who is not None:
        if opt.intent == "work":
            r["values"] = (0.9 * who.holds("craft") + 0.7 * who.holds("security")
                           + 0.5 * who.holds("standing")
                           - 0.8 * who.holds("freedom"))
        elif opt.intent == "socialize":
            r["values"] = (0.8 * who.holds("belonging") + 0.6 * who.holds("pleasure")
                           + (0.5 * who.holds("family") if opt.person
                              in u.family.kin() else 0.0))
        elif opt.intent in ("wander", "seek"):
            r["values"] = 1.0 * who.holds("freedom") - 0.4 * who.holds("security")
        elif opt.intent == "eat":
            r["values"] = 0.5 * who.holds("pleasure")
        elif opt.intent == "sleep":
            r["values"] = 0.3 * who.holds("security")
        elif opt.intent == "rob":
            r["values"] = -1.6 * who.holds("fairness") - 0.6 * who.holds("standing")
        if r.get("values", 0.0) == 0.0:
            r.pop("values", None)
        if opt.intent == "work" and u.dependents_worry(sim.clock.tick) > 0.2:
            r["for them"] = 1.1 * who.holds("family") \
                * u.dependents_worry(sim.clock.tick)

    # --- what I am trying to bring about -----------------------------------
    pull = goals_mod.toward(u, opt)
    if abs(pull) > 0.01:
        r["what I'm after"] = pull

    # --- what has my attention ---------------------------------------------
    # Only what won the workspace gets to push the decision around. A drive a
    # unit is not attending to still exists and still decays; it just is not
    # what this hour is about.
    att = u.workspace.current
    if att is not None:
        if att.kind == "drive" and att.detail.get("drive") == "hunger" \
                and opt.intent == "eat":
            r["on my mind"] = 1.1 * att.salience
        elif att.kind == "drive" and att.detail.get("drive") == "fatigue" \
                and opt.intent in ("sleep", "rest"):
            r["on my mind"] = 1.0 * att.salience
        elif att.kind == "drive" and att.detail.get("drive") == "loneliness" \
                and opt.intent == "socialize":
            r["on my mind"] = 1.0 * att.salience
        elif att.kind == "social" and opt.person == att.detail.get("who"):
            r["on my mind"] = 0.9 * att.salience
        elif att.kind == "pain" and opt.intent in ("rest", "sleep"):
            r["on my mind"] = 0.8 * att.salience

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
    if u.status == "unemployed":
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


def decide(sim, u):
    """The whole decision: fast path first, slow path only if it has to.

    This is the entry point the simulation calls. Most hours it returns a
    habit without ever enumerating an option.
    """
    # deliberate recall, which rebuilds what it touches
    u.memory.recall(sim.clock.tick, k=2, rehearse=True,
                    mood=u.affect.intensity, stress=u.affect.stress,
                    district=u.district)
    ep = u.memory.intrude(sim.clock.tick, u.rng, u.affect.trauma)
    if ep is not None:
        u.affect.feel(ep.emotions, scale=0.35)
        u.rumination += 1
    dwell(u)
    u.view = perception.perceive(sim, u)

    # Attention first: one thing wins the hour, and only that is broadcast.
    attended = u.workspace.compete(workspace.bid(u, u.view))

    key = dualprocess.situation(u, u.view)
    u.situation = key
    reason = dualprocess.needs_thought(u, u.view, key)
    worn_out = bool(reason) and u.effort.available() < 0.12
    if worn_out:
        reason = ""                 # too worn down to think, whatever the reason
    if not reason:
        habit = u.habits.get(key)
        if habit is not None and u.habits.confidence(key) >= 0.14:
            u.thought = False
            # set every time: carrying the previous decision's reason forward
            # made the inspector report a unit as having stopped to think
            # about something it had long since stopped thinking about
            u.why_thought = ("too tired to think" if worn_out
                             else "did what it always does")
            opt = Option(habit.intent, habit.venue)
            opt.reasons = {"habit": u.habits.confidence(key)}
            u.considered = [opt]
            u.chose = opt
            return opt
        reason = "nothing to fall back on"

    u.thought = True
    u.why_thought = reason
    u.effort.spend()
    return deliberate(sim, u)


def deliberate(sim, u):
    """The slow path: options, evaluation, choice."""
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

    # Satisficing: an option far behind the best one is not weighed at all.
    # Without this the softmax leaked a few percent onto every option every
    # hour, which is harmless for whether to rest or wander and absurd for
    # robbery -- units committed 339 of them in two months while solvent,
    # not because it scored well but because nothing scores zero.
    top = max(o.score for o in opts)
    opts = [o for o in opts if o.score >= top - 2.5]

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
    u.view = perception.perceive(sim, u)
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
    # what this happened on the back of: the last thing the unit did, which
    # is how an episode records a because rather than only a what
    because = u.chose.label() if u.chose is not None else ""
    u.memory.encode(Episode(tick=sim.clock.tick, kind=kind, text=text,
                            place=place, district=district, people=tuple(people),
                            emotions=emotions, because=because), novelty=novelty)
    return emotions
