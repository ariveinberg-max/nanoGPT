"""Tests for the prototype. Run with: python -m simulacron.test_simulacron"""

import sys

import numpy as np

from . import world
from .brain import INTENTS, Learner, Policy
from .link import LinkSession
from .sim import Simulation
from .unit import N_OBS, Unit

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def test_world():
    print("world")
    check("districts are all reachable",
          all(world.travel_ticks(a, b) < 50
              for a in world.DISTRICTS for b in world.DISTRICTS))
    check("travel is symmetric",
          all(world.travel_ticks(a, b) == world.travel_ticks(b, a)
              for a in world.DISTRICTS for b in world.DISTRICTS))
    check("every district has a venue",
          all(world.venues_in(d) for d in world.DISTRICTS))
    check("every unit can find work, food, a bed and company",
          all(world.venues_of(k) for k in ("home", "work", "food", "social")))
    c = world.Clock(tick=world.TICKS_PER_DAY * 5)
    check("weekends land on days 5 and 6", not c.is_workday and c.weekday == "Saturday",
          c.weekday)
    night = world.VENUES_BY_KEY["club_alabam"]
    check("venues open across midnight", night.open_at(23) and night.open_at(1)
          and not night.open_at(12))


def test_learner_finds_reward():
    print("brain")
    rng = np.random.default_rng(0)
    p = Policy(4, rng=rng)
    l = Learner(p, lr=0.05, gamma=0.0, horizon=48, entropy=0.0, entropy_floor=0.0)
    obs = np.array([1.0, 0.0, 0.0, 0.0])
    target = 2
    for _ in range(3000):
        a, rec = p.act(obs)
        l.record(rec, 1.0 if a == target else 0.0)
    prob = p.forward(obs)[1][target]
    check("policy gradient concentrates on the rewarded intent", prob > 0.9,
          f"p={prob:.3f}")


def test_unit():
    print("unit")
    u = Unit.spawn(np.random.default_rng(7))
    obs = u.observe(world.Clock())
    check("observation width matches the declared layout", len(obs) == N_OBS)
    check("observations are finite", np.all(np.isfinite(obs)))
    for _ in range(500):
        u.decay()
    check("drives stay bounded under decay",
          0 <= u.hunger <= 1 and 0 <= u.fatigue <= 1 and 0 <= u.loneliness <= 1)
    u.remember(0, "the same thing")
    u.remember(1, "the same thing")
    check("memory does not stutter", len(u.memories) == 1)


def test_sim_runs_unattended():
    print("simulation")
    s = Simulation(n_units=8, seed=5)
    s.run_days(20)
    check("clock advanced", s.clock.day == 20, s.clock.day)
    check("every unit is somewhere or travelling",
          all(u.location_key or u.travelling for u in s.units))
    check("no unit holds an unknown intent",
          all(u.intent in INTENTS for u in s.units))
    check("funds never go negative", all(u.funds >= -1e-9 for u in s.units),
          min(u.funds for u in s.units))
    check("every unit accumulated experience",
          all(u.learner.reward_history for u in s.units))


def test_units_learn():
    """The load-bearing claim: they are self-learning, not scripted."""
    print("learning")
    s = Simulation(n_units=16, seed=3)
    s.run_days(30)
    early = s.stats()["reward"]
    s.run_days(400)
    late = s.stats()["reward"]
    check("population reward improves with experience", late > early + 0.2,
          f"{early:+.3f} -> {late:+.3f}")

    worked_weekday, worked_weekend = 0, 0
    for _ in range(7):
        counter = [0]
        workday = s.clock.is_workday
        s.run_days(1, on_tick=lambda sim: counter.__setitem__(
            0, counter[0] + sum(1 for u in sim.units if u.worked_last)))
        if workday:
            worked_weekday += counter[0]
        else:
            worked_weekend += counter[0]
    check("units learned the working week",
          worked_weekday > 0 and worked_weekend == 0,
          f"weekday={worked_weekday} weekend={worked_weekend}")


def test_link():
    print("link")
    s = Simulation(n_units=6, seed=9)
    s.run_days(10)
    u = s.units[0]
    before = len(u.learner.reward_history)
    link = LinkSession(s, u)
    link.jack_in()
    check("the unit is carrying an operator", u.linked)
    s.run_days(1)
    check("a linked unit stops learning on its own",
          len(u.learner.reward_history) == before)
    check("look() describes somewhere", "1937" in link.look())
    msg = link.jack_out()
    check("link releases the body", not u.linked and "SEVERED" in msg)
    check("the unit noticed the missing hours", u.dissonance > 0,
          f"{u.dissonance:.3f}")
    check("the gap is in its memory",
          any("cannot account" in m.text for m in u.memories))

    second = LinkSession(s, u)
    second.jack_in()
    try:
        LinkSession(s, u).jack_in()
        check("double occupancy is refused", False)
    except RuntimeError:
        check("double occupancy is refused", True)
    second.jack_out()


def test_determinism():
    print("determinism")
    a = Simulation(n_units=6, seed=42); a.run_days(15)
    b = Simulation(n_units=6, seed=42); b.run_days(15)
    check("same seed, same prototype",
          [u.funds for u in a.units] == [u.funds for u in b.units])


def main():
    for t in (test_world, test_learner_finds_reward, test_unit,
              test_sim_runs_unattended, test_units_learn, test_link,
              test_determinism):
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
