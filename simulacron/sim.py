"""The system itself.

It does not need a user to interact with it to function. Start it and it runs:
units wake, ride the Metro, work, eat, fall out with each other, get sick, get
into debt, and quietly get better at being themselves. An operator may link in,
but the simulation does not notice the difference except where a linked unit
leaves a hole in its own memory.

The loop here is deliberately thin. It decides nothing: it asks `cognition`
what each unit wants to do, carries it out, and hands the consequences back as
events for the unit to appraise, feel and remember.
"""

import numpy as np

from . import cognition, lifecourse, world
from .affect import Appraisal
from .brain import INTENTS, Learner, Policy
from .unit import N_OBS, Unit, circadian, pick_job

# A unit commits to an intent for an hour at a time. Deciding afresh every
# quarter hour produced twitching sleepwalkers who spent all day on the
# freeway; an hour is about the grain a person actually plans at.
DECISION_TICKS = world.TICKS_PER_HOUR

RENT_PER_DAY = 0.34          # as a fraction of DAILY_COST
ILLNESS_BASE = 0.004         # per day, before stress and hunger


class Simulation:
    def __init__(self, n_units=24, seed=2010, log=None, shared_mind=False,
                 deliberate=True):
        """`shared_mind` puts every unit's habit layer on one set of weights.

        Off by default, and kept only because it is the obvious thing to reach
        for and worth being able to reproduce: units differ in where they live,
        where they work and what they want, and one set of weights serving all
        of them collapsed onto a single intent for every hour of the day.
        """
        self.rng = np.random.default_rng(seed)
        self.clock = world.Clock()
        self.shared_mind = shared_mind
        # `deliberate=False` runs the same bodies in the same world choosing
        # uniformly at random from the same options. It is the control the
        # cognitive layer is measured against: against a grind like this one,
        # holding steady is the achievement, and improvement over time is the
        # wrong thing to look for.
        self.deliberating = deliberate
        self.units = [Unit.spawn(self.rng) for _ in range(n_units)]
        if shared_mind:
            substrate = Policy(N_OBS, rng=self.rng)
            for u in self.units:
                u.policy = substrate
                u.learner = Learner(substrate, share=n_units)
        self.by_name = {}
        self.log = log if log is not None else []
        self.anomalies = []
        self.dead = []              # (tick, unit, cause), kept for the record
        self.born = 0
        self._target = n_units
        for u in self.units:
            self._name_uniquely(u)

    # ------------------------------------------------------------------ core
    def step(self):
        """Advance the whole prototype by fifteen minutes."""
        clock = self.clock
        crowd = self._crowd_by_venue()
        for u in self.units:
            if u.travelling:
                u.activity = "in transit"
                self._continue_trip(u, crowd)
                continue
            if u.linked:
                u.decay(clock)          # the operator is driving
                continue
            if u.hold_left <= 0:
                self._decide(u, crowd)
            else:
                self._perform(u, crowd, first=False)
            u.decay(clock)
            self._settle(u)

        if clock.tick % world.TICKS_PER_DAY == 0 and clock.tick:
            for u in list(self.units):
                self._daily(u)
            self._reap()
            self._arrivals()
        self._check_anomalies()
        clock.advance()

    def run(self, ticks, on_tick=None):
        for _ in range(ticks):
            self.step()
            if on_tick:
                on_tick(self)

    def run_days(self, days, on_tick=None):
        self.run(days * world.TICKS_PER_DAY, on_tick)

    # ----------------------------------------------------------- deliberation
    def _decide(self, u, crowd):
        obs = u.observe(self.clock)
        action, record = u.policy.act(obs)          # habit layer, for learning
        if self.deliberating:
            option = cognition.deliberate(self, u)
        else:
            option = cognition.flail(self, u)
        u.intent = option.intent
        u.target = option.venue
        u.pending_record = record
        u.pending_reward = 0.0
        u.hold_left = DECISION_TICKS
        if option.venue and option.venue != u.location_key:
            self._send(u, option.venue)
        else:
            self._perform(u, crowd, first=True)

    def _settle(self, u):
        if u.pending_record is None:
            return
        u.pending_reward += u.reward()
        if u.travelling:
            return                      # the hour starts when the unit arrives
        u.hold_left -= 1
        if u.hold_left <= 0:
            u.learner.record(u.pending_record, u.pending_reward / DECISION_TICKS)
            u.selfmodel.outcome(u.intent, u.pending_reward / DECISION_TICKS)
            u.pending_record = None
            u.pending_reward = 0.0

    def _continue_trip(self, u, crowd):
        u.travel_left -= 1
        if u.travel_left == 0:
            u.location_key = u.travel_to
            u.district = world.VENUES_BY_KEY[u.travel_to].district
            u.travel_to = ""
            if not u.linked:
                self._perform(u, crowd, first=True)
        u.decay(self.clock)
        if not u.linked:
            self._settle(u)

    # -------------------------------------------------------------- execution
    def _perform(self, u, crowd, first=True):
        """Carry out what the unit decided, and hand back what happened."""
        clock = self.clock
        here = u.location
        u.worked_last = False
        u.earned = 0.0
        u.activity = "idle"
        intent = u.intent

        if intent == "sleep":
            if u.location_key == u.home_key:
                u.activity = "asleep"
                u.fatigue = max(0.0, u.fatigue - 0.034 * circadian(clock))
                u.loneliness = min(1.0, u.loneliness + 0.002)

        elif intent == "eat":
            if here and here.kind == "food" and here.open_at(clock.hour) and u.funds >= here.cost:
                u.activity = "eating"
                if first:
                    u.funds -= here.cost
                    u.hunger = max(0.0, u.hunger - 0.85)
                    cognition.experience(
                        self, u, "ate", f"ate at {here.name}",
                        Appraisal(valence=0.35, agency="self"), place=here.key)
            elif u.location_key == u.home_key and u.funds >= world.HOME_MEAL_COST:
                u.activity = "eating"
                if first:
                    u.funds -= world.HOME_MEAL_COST
                    u.hunger = max(0.0, u.hunger - 0.55)
            elif first:
                cognition.experience(
                    self, u, "turned_away", "wanted to eat and could not",
                    Appraisal(valence=-0.35 * (0.4 + u.hunger), control=0.25),
                    place=u.location_key)

        elif intent == "work":
            w = u.workplace
            if (u.employed and u.location_key == u.work_key
                    and w.open_at(clock.hour) and clock.is_workday):
                u.activity = "working"
                u.worked_last = True
                pay = w.wage / world.TICKS_PER_HOUR * min(1.0, u.capacity)
                u.funds += pay
                u.earned = pay
                u.fatigue = min(1.0, u.fatigue + 0.010)
                u.loneliness = max(0.0, u.loneliness - 0.008)
                u.week_ticks += 1
                if first:
                    u.selfmodel.did("work", True)
                    u.selfmodel.endorse("worker", 0.004)
            elif first:
                u.selfmodel.did("work", False)
                cognition.experience(
                    self, u, "shut_out", "turned up and could not work",
                    Appraisal(valence=-0.25, control=0.2), place=u.location_key)

        elif intent == "socialize":
            if here and here.kind in ("social", "civic") and here.open_at(clock.hour) \
                    and u.funds >= here.cost:
                u.activity = "out"
                company = [o for o in crowd.get(u.location_key, ()) if o is not u]
                if first:
                    u.funds -= here.cost
                u.loneliness = max(0.0, u.loneliness
                                   - (0.09 + 0.03 * min(len(company), 4)))
                if company:
                    other = company[0]
                    rel = u.social.met(other.name, clock.tick,
                                       quality=0.5 + 0.5 * u.traits["sociability"])
                    other.social.met(u.name, clock.tick, quality=0.4)
                    if first:
                        # showing up for someone is what moves trust
                        u.social.treated(other.name, clock.tick, valence=0.22)
                        other.social.treated(u.name, clock.tick, valence=0.22)
                        u.selfmodel.did("social", True)
                        cognition.experience(
                            self, u, "company", f"spent time with {other.name} at {here.name}",
                            Appraisal(valence=0.3 + 0.5 * rel.closeness(),
                                      agency="other", other=other.name),
                            place=here.key, people=(other.name,))
                elif first:
                    u.selfmodel.did("social", False)
                    u.loneliness = min(1.0, u.loneliness + 0.01)
                    # Only worth remembering when it stung. Recording every
                    # quiet hour out filled memory with identical nothings.
                    if u.loneliness > 0.5:
                        cognition.experience(
                            self, u, "alone", f"sat alone at {here.name}",
                            Appraisal(valence=-0.35 * u.traits["sociability"],
                                      control=0.4, irreversible=0.2),
                            place=here.key)

        elif intent == "rest":
            u.activity = "resting"
            u.fatigue = max(0.0, u.fatigue - 0.016)

        elif intent == "errand":
            if u.funds >= world.ERRAND_COST:
                u.activity = "errands"
                if first:
                    u.funds -= world.ERRAND_COST
                    u.hunger = max(0.0, u.hunger - 0.08)
                    u.job_search += 0.4

        elif intent == "wander":
            u.activity = "out"
            u.fatigue = min(1.0, u.fatigue + 0.004)

        elif intent == "seek":
            u.activity = "out"
            u.fatigue = min(1.0, u.fatigue + 0.005)
            u.job_search += 0.6
            if first:
                self._look_for_the_edge(u)

    def _look_for_the_edge(self, u):
        """A unit that goes looking finds where the world stops.

        Which seam it hits depends on where it went looking, so a unit that
        keeps walking west keeps finding the same one, and repetition is what
        erodes the account rather than any single shock.
        """
        r = u.rng.random()
        if u.district == world.EDGE_DISTRICT and r < 0.10:
            cognition.probe_world(self, u, "edge")
        elif u.location_key == "observatory" and self.clock.hour >= 19 and r < 0.22:
            cognition.probe_world(self, u, "sky")
        elif world.travel_ticks(u.home.district, u.district) >= 4 and r < 0.03:
            cognition.probe_world(self, u, "horizon")
        elif r < 0.005:
            cognition.probe_world(
                self, u, "elsewhere",
                str(u.rng.choice(u.worldview.places)))

    # ------------------------------------------------------------- the grind
    def _daily(self, u):
        """Rent, illness, and whether there is still a job to go to."""
        u.age += 1.0 / lifecourse.DAYS_PER_YEAR
        u.refresh_stage()
        if u.rng.random() < lifecourse.natural_risk(u.age):
            u.health = 0.0          # reaped at the end of the day
        if u.employed and not lifecourse.can_work(u.age):
            u.employed = False
            u.selfmodel.endorse("worker", -0.2)
            cognition.experience(
                self, u, "retired", "too old for the work now",
                Appraisal(valence=-0.35, control=0.2, irreversible=0.8))

        rent = RENT_PER_DAY * world.DAILY_COST
        if u.funds >= rent:
            u.funds -= rent
        else:
            short = rent - u.funds
            u.funds = 0.0
            u.debt += short
            cognition.experience(
                self, u, "short", "could not make the rent",
                Appraisal(valence=-0.45, control=0.2, irreversible=0.3))
        if u.debt > 0:
            spare = u.funds - 2 * rent
            if spare > 0:
                paid = min(spare, u.debt)
                u.funds -= paid
                u.debt -= paid
                if u.debt <= 0.01:
                    u.debt = 0.0
                    cognition.experience(
                        self, u, "cleared", "cleared what was owed",
                        Appraisal(valence=0.5, agency="self", norm=0.5))
            u.debt *= 1.004
            u.selfmodel.endorse("provider", -0.004)
        else:
            u.selfmodel.endorse("provider", 0.004)

        was = u.peril()
        if u.hunger > 0.88:
            u.health = max(0.0, u.health - 0.055)
            if u.health < 0.5:
                cognition.experience(
                    self, u, "wasting", "another day without eating",
                    Appraisal(valence=-0.7, anticipated=True, certainty=0.7,
                              control=0.3, irreversible=0.4))
        if u.pain > 0.5 and u.health < 0.6:
            u.health = max(0.0, u.health - 0.02)

        risk = (ILLNESS_BASE * (1 + 1.6 * u.affect.stress + 1.2 * u.hunger)
                * (1.0 + 4.0 * u.frailty))
        if u.health > 0.4 and u.rng.random() < risk:
            u.health = max(0.2, u.health - float(u.rng.uniform(0.2, 0.45)))
            u.pain = min(1.0, u.pain + float(u.rng.uniform(0.25, 0.6)))
            cognition.experience(
                self, u, "ill", "woke up ill",
                Appraisal(valence=-0.55, control=0.15, irreversible=0.2))

        u.selfmodel.survived(was, u.peril())
        if u.social.people and max(
                (r.closeness() for r in u.social.people.values()), default=0) > 0.45:
            u.selfmodel.endorse("friend", 0.01)

        if u.employed:
            worked = u.week_ticks
            if self.clock.day % 7 == 0:
                # The threshold sat right at what units actually work, so
                # almost everyone drew for dismissal every week and the city
                # churned through jobs. It marks genuine absence now.
                if worked < 4 * world.TICKS_PER_HOUR and u.rng.random() < 0.25:
                    u.employed = False
                    u.selfmodel.endorse("worker", -0.3)
                    u.selfmodel.did("work", False)
                    cognition.experience(
                        self, u, "lost_job", f"let go from {u.workplace.name}",
                        Appraisal(valence=-0.85, control=0.15, irreversible=0.6))
                    self.log.append(f"[{self.clock.stamp()}] {u.name} lost their job.")
                u.week_ticks = 0
        else:
            effort = u.job_search
            chance = 0.02 + 0.10 * min(1.0, effort / 6.0) * u.selfmodel.can("work")
            if u.rng.random() < chance:
                u.employed = True
                u.work_key = pick_job(u.rng, u.home.district).key
                u.selfmodel.endorse("worker", 0.25)
                cognition.experience(
                    self, u, "found_job", f"took work at {u.workplace.name}",
                    Appraisal(valence=0.7, agency="self", norm=0.5))
                self.log.append(f"[{self.clock.stamp()}] {u.name} found work "
                                f"at {u.workplace.name}.")
            u.job_search = effort * 0.5
            u.selfmodel.endorse("outsider", 0.01)

    def _name_uniquely(self, u):
        """Names are drawn from a small pool, so collisions are routine. They
        used to corrupt the per-unit bookkeeping that was keyed by name."""
        if u.name in self.by_name:
            n = 2
            while f"{u.name} ({n})" in self.by_name:
                n += 1
            u.name = f"{u.name} ({n})"
        self.by_name[u.name] = u

    # ------------------------------------------------------------ mortality
    def _reap(self):
        """Nothing in this simulation mattered until this function existed."""
        for u in list(self.units):
            if not u.dying():
                continue
            cause = ("old age" if u.age >= 70 and u.hunger < 0.7 and u.pain < 0.4
                     else "starvation" if u.hunger > 0.85 else
                     "illness")
            self._die(u, cause)

    def _die(self, u, cause):
        self.units.remove(u)
        self.dead.append((self.clock.tick, u, cause))
        self.by_name.pop(u.name, None)
        self.log.append(f"[{self.clock.stamp()}] {u.name} died of {cause}, "
                        f"aged {u.age:.0f}.")
        if u.linked:
            self.log.append(f"[{self.clock.stamp()}] LINK LOST -- the operator's "
                            f"body was riding {u.name}.")
        for other in self.units:
            rel = other.social.people.get(u.name)
            if rel is None or rel.familiarity < 0.12:
                continue
            closeness = rel.closeness()
            cognition.experience(
                self, other, "bereaved", f"{u.name} died",
                Appraisal(valence=-0.45 - 0.55 * closeness, control=0.0,
                          irreversible=1.0), people=(u.name,))
            rel.grudge = 0.0

    def _arrivals(self):
        """People keep coming to this city. Until there are births, this is why
        the prototype does not simply empty out."""
        if len(self.units) >= self._target:
            return
        if self.rng.random() < 0.25:
            u = Unit.spawn(self.rng)
            self.units.append(u)
            self._name_uniquely(u)

    # ------------------------------------------------------------- utilities
    def _crowd_by_venue(self):
        crowd = {}
        for u in self.units:
            if not u.travelling and u.location_key:
                crowd.setdefault(u.location_key, []).append(u)
        return crowd

    def district_population(self):
        out = {}
        for u in self.units:
            if not u.travelling:
                out.setdefault(u.district, []).append(u)
        return out

    def _send(self, u, key):
        if not key or key == u.location_key:
            return
        dest = world.VENUES_BY_KEY[key]
        u.travel_to = key
        u.travel_left = max(1, world.travel_ticks(u.district, dest.district))
        u.location_key = ""

    def _note(self, u, text):
        u.remember(self.clock.tick, text)

    def _check_anomalies(self):
        for u in self.units:
            if u.dissonance >= 1.0 and not any(a[1] is u for a in self.anomalies):
                self.anomalies.append((self.clock.tick, u))
                self.log.append(
                    f"[{self.clock.stamp()}] ANOMALY: {u.name} is asking questions "
                    f"about the shape of the world.")

    # ------------------------------------------------------------- reporting
    def stats(self):
        us = self.units
        if not us:
            return {"day": self.clock.day, "alive": 0, "dead": len(self.dead)}
        return {
            "day": self.clock.day,
            "wellbeing": float(np.mean([u.wellbeing() for u in us])),
            "reward": float(np.mean([u.learner.recent_reward() for u in us])),
            "funds": float(np.mean([u.funds for u in us])),
            "debt": float(np.mean([u.debt for u in us])),
            "hunger": float(np.mean([u.hunger for u in us])),
            "fatigue": float(np.mean([u.fatigue for u in us])),
            "stress": float(np.mean([u.affect.stress for u in us])),
            "trauma": float(np.mean([u.affect.trauma for u in us])),
            "peril": float(np.mean([u.peril() for u in us])) if us else 0.0,
            "alive": len(us),
            "dead": len(self.dead),
            "unemployed": sum(1 for u in us if not u.employed),
            "ill": sum(1 for u in us if u.health < 0.9),
            "friends": float(np.mean([u.social.known() for u in us])),
            "anomalies": len(self.anomalies),
        }
