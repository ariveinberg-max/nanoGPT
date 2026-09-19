"""The system itself.

It does not need a user to interact with it to function. Start it and it runs:
units wake, ride the Red Cars, work, eat, drink, and quietly get better at
being themselves. An operator may link in, but the simulation does not notice
the difference except where a linked unit leaves a hole in its own memory.
"""

import numpy as np

from . import world
from .brain import INTENTS
from .unit import Unit

# A unit commits to an intent for an hour at a time. Deciding afresh every
# quarter hour produced twitching sleepwalkers who spent all day on the Red
# Cars; an hour is about the grain a person actually plans at.
DECISION_TICKS = world.TICKS_PER_HOUR


class Simulation:
    def __init__(self, n_units=24, seed=1937, log=None):
        self.rng = np.random.default_rng(seed)
        self.clock = world.Clock()
        self.units = [Unit.spawn(self.rng) for _ in range(n_units)]
        self.by_name = {u.name: u for u in self.units}
        self.log = log if log is not None else []
        self.anomalies = []

    # ------------------------------------------------------------------ core
    def step(self):
        """Advance the whole prototype by fifteen minutes."""
        clock = self.clock
        crowd = self._crowd_by_venue()
        for u in self.units:
            if u.travelling:
                self._continue_trip(u, crowd)
                continue

            if u.linked:
                # the operator is driving; the unit's own policy is suspended
                u.decay()
                continue

            if u.hold_left <= 0:
                self._decide(u, crowd)
            else:
                self._resolve(u, u.intent, crowd, may_travel=False)
            u.decay()
            self._settle(u)

        self._check_anomalies()
        clock.advance()

    def _decide(self, u, crowd):
        """Sample a fresh intent and commit to it for the next hour."""
        obs = u.observe(self.clock)
        action, step_record = u.policy.act(obs)
        u.intent = INTENTS[action]
        u.pending_record = step_record
        u.pending_reward = 0.0
        u.hold_left = DECISION_TICKS
        self._resolve(u, u.intent, crowd, may_travel=True)

    def _settle(self, u):
        """Charge the tick against the decision that caused it."""
        if u.pending_record is None:
            return
        u.pending_reward += u.reward()
        if u.travelling:
            return                      # the hour starts when the unit arrives
        u.hold_left -= 1
        if u.hold_left <= 0:
            u.learner.record(u.pending_record, u.pending_reward / DECISION_TICKS)
            u.pending_record = None
            u.pending_reward = 0.0

    def _continue_trip(self, u, crowd):
        """A unit on a Red Car stays committed to the errand it boarded for."""
        u.travel_left -= 1
        arrived = u.travel_left == 0
        if arrived:
            u.location_key = u.travel_to
            u.district = world.VENUES_BY_KEY[u.travel_to].district
            u.travel_to = ""
            if not u.linked:
                self._resolve(u, u.intent, crowd, may_travel=False)
        u.decay()
        if not u.linked:
            self._settle(u)

    def run(self, ticks, on_tick=None):
        for _ in range(ticks):
            self.step()
            if on_tick:
                on_tick(self)

    def run_days(self, days, on_tick=None):
        self.run(days * world.TICKS_PER_DAY, on_tick)

    # ------------------------------------------------------------- resolution
    def _crowd_by_venue(self):
        crowd = {}
        for u in self.units:
            if not u.travelling and u.location_key:
                crowd.setdefault(u.location_key, []).append(u)
        return crowd

    def _resolve(self, u, intent, crowd, may_travel=True):
        """Carry out an intent here, or set off for somewhere it is possible."""
        clock = self.clock
        here = u.location
        u.worked_last = False
        u.earned = 0.0

        def go(key):
            if may_travel:
                self._send(u, key)

        if intent == "sleep":
            if u.location_key == u.home_key:
                u.fatigue = max(0.0, u.fatigue - 0.034)
                u.loneliness = min(1.0, u.loneliness + 0.002)
            else:
                go(u.home_key)

        elif intent == "eat":
            if here and here.kind == "food" and here.open_at(clock.hour) and u.funds >= here.cost:
                u.funds -= here.cost
                u.hunger = max(0.0, u.hunger - 0.55)
                u.mood = min(1.0, u.mood + 0.04)
                self._note(u, f"ate at {here.name}")
            elif u.location_key == u.home_key and u.funds >= 0.08:
                u.funds -= 0.08
                u.hunger = max(0.0, u.hunger - 0.30)
            else:
                go(self._nearest(u, "food", affordable=True) or u.home_key)

        elif intent == "work":
            w = u.workplace
            if u.location_key == u.work_key and w.open_at(clock.hour) and clock.is_workday:
                pay = w.wage / world.TICKS_PER_HOUR
                u.funds += pay
                u.earned = pay
                u.fatigue = min(1.0, u.fatigue + 0.010)
                u.loneliness = max(0.0, u.loneliness - 0.008)
                u.worked_last = True
            elif clock.is_workday and w.open_at(clock.hour):
                go(u.work_key)
            else:
                # a shut door, or a Sunday: the hour is spent standing about
                u.fatigue = min(1.0, u.fatigue + 0.005)
                u.mood = max(0.0, u.mood - 0.02)

        elif intent == "socialize":
            if (here and here.kind in ("social", "civic")
                    and here.open_at(clock.hour) and u.funds >= here.cost):
                u.funds -= here.cost
                company = max(0, len(crowd.get(u.location_key, [])) - 1)
                u.loneliness = max(0.0, u.loneliness - (0.25 + 0.10 * min(company, 4)))
                u.mood = min(1.0, u.mood + 0.05)
                if company:
                    other = next(o for o in crowd[u.location_key] if o is not u)
                    self._note(u, f"ran into {other.name} at {here.name}")
                    self._note(other, f"ran into {u.name} at {here.name}")
            else:
                go(self._nearest(u, "social", affordable=True)
                   or self._nearest(u, "civic"))

        elif intent == "wander":
            u.fatigue = min(1.0, u.fatigue + 0.004)
            u.mood = min(1.0, u.mood + 0.01 * u.traits["restlessness"])
            if may_travel and u.rng.random() < 0.25:
                d = str(u.rng.choice(world.DISTRICTS))
                targets = world.venues_in(d, "civic") or world.venues_in(d, "social")
                if targets:
                    go(targets[0].key)

        elif intent == "rest":
            u.fatigue = max(0.0, u.fatigue - 0.016)

        elif intent == "errand":
            if u.funds >= 0.15:
                u.funds -= 0.15
                u.hunger = max(0.0, u.hunger - 0.08)
                u.mood = min(1.0, u.mood + 0.01)
            else:
                u.mood = max(0.0, u.mood - 0.02)

        elif intent == "seek":
            # Probing the far edges of the prototype. Harmless, until it isn't.
            u.fatigue = min(1.0, u.fatigue + 0.005)
            if may_travel and u.rng.random() < 0.20:
                far = max(world.DISTRICTS,
                          key=lambda d: world.travel_ticks(u.district, d))
                go(world.venues_in(far)[0].key)
            if u.district == "Venice" and u.rng.random() < 0.06:
                u.dissonance = min(2.0, u.dissonance + 0.04)
                self._note(u, "stood at the edge of town and could not say what lay past it")

    # ------------------------------------------------------------- utilities
    def _send(self, u, key):
        if not key or key == u.location_key:
            return
        dest = world.VENUES_BY_KEY[key]
        t = world.travel_ticks(u.district, dest.district)
        u.travel_to = key
        u.travel_left = max(1, t)
        u.location_key = ""

    def _nearest(self, u, kind, affordable=False):
        cands = [v for v in world.venues_of(kind)
                 if v.open_at(self.clock.hour)
                 and (not affordable or v.cost <= u.funds)]
        if not cands:
            return None
        return min(cands, key=lambda v: (world.travel_ticks(u.district, v.district), v.cost)).key

    def _note(self, u, text):
        u.remember(self.clock.tick, text)

    def _check_anomalies(self):
        for u in self.units:
            if u.dissonance >= 1.0 and not any(a[1] is u for a in self.anomalies):
                self.anomalies.append((self.clock.tick, u))
                self.log.append(
                    f"[{self.clock.stamp()}] ANOMALY: {u.name} is asking questions "
                    f"about the shape of the world."
                )

    # ------------------------------------------------------------- reporting
    def stats(self):
        return {
            "day": self.clock.day,
            "wellbeing": float(np.mean([u.wellbeing() for u in self.units])),
            "reward": float(np.mean([u.learner.recent_reward() for u in self.units])),
            "funds": float(np.mean([u.funds for u in self.units])),
            "hunger": float(np.mean([u.hunger for u in self.units])),
            "fatigue": float(np.mean([u.fatigue for u in self.units])),
            "employed_now": sum(1 for u in self.units if u.location_key == u.work_key),
            "anomalies": len(self.anomalies),
        }

    def intent_profile(self):
        """How the population spends its day, by hour -- the learned rhythm."""
        counts = {}
        for u in self.units:
            counts[u.intent] = counts.get(u.intent, 0) + 1
        return counts
