"""Tests for the prototype. Run with: python -m simulacron.test_simulacron"""

import sys

import numpy as np

from . import world
from .affect import Affect, Appraisal
from .brain import INTENTS, Learner, Policy
from .cognition import deliberate, evaluate, generate
from .link import LinkSession
from .memory import Episode, Memory
from .selfmodel import SelfModel
from .sim import Simulation
from .social import Social
from .unit import N_OBS, Unit
from .worldview import Worldview, inherit

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
    night = next(v for v in world.VENUES if v.opens > v.closes)
    check("venues open across midnight",
          night.open_at(23) and night.open_at(night.closes - 1)
          and not night.open_at(night.opens - 1), night.name)
    check("a day's costs are set for the era", world.DAILY_COST > 0)
    check("nobody works below the 2010 California minimum",
          all(v.wage >= 8.00 for v in world.venues_of("work")),
          min(v.wage for v in world.venues_of("work")))


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


def test_circadian():
    print("circadian")
    from .unit import circadian
    night = circadian(world.Clock(tick=3 * world.TICKS_PER_HOUR))
    noon = circadian(world.Clock(tick=15 * world.TICKS_PER_HOUR))
    check("sleep pressure peaks in the small hours", night > 1.4 > 0.7 > noon,
          f"3am={night:.2f} 3pm={noon:.2f}")


def test_unit():
    print("unit")
    u = Unit.spawn(np.random.default_rng(7))
    obs = u.observe(world.Clock())
    check("observation width matches the declared layout", len(obs) == N_OBS)
    check("observations are finite", np.all(np.isfinite(obs)))
    clock = world.Clock()
    for _ in range(500):
        u.decay(clock)
        clock.advance()
    check("drives stay bounded under decay",
          0 <= u.hunger <= 1 and 0 <= u.fatigue <= 1 and 0 <= u.loneliness <= 1)
    check("pain counts against wellbeing",
          (lambda a, b: b < a)(u.wellbeing(), (setattr(u, "pain", 0.5), u.wellbeing())[1]))


def test_appraisal():
    """The point of appraisal: same badness, different emotion."""
    print("appraisal")
    robbed = Appraisal(valence=-0.8, agency="other", control=0.2, norm=-1.0).emotions()
    laid_off = Appraisal(valence=-0.8, agency="circumstance", control=0.15,
                         irreversible=0.8).emotions()
    caught = Appraisal(valence=-0.6, agency="self", norm=-0.9, public=1.0,
                       harmed_other=0.5).emotions()
    dark = Appraisal(valence=-0.7, anticipated=True, certainty=0.4,
                     control=0.2).emotions()
    check("being wronged produces anger",
          max(robbed, key=robbed.get) == "anger", robbed)
    check("being laid off produces sadness, not anger",
          max(laid_off, key=laid_off.get) == "sadness"
          and laid_off.get("anger", 0) == 0, laid_off)
    check("one's own fault produces shame and guilt",
          caught.get("shame", 0) > 0.3 and caught.get("guilt", 0) > 0.3, caught)
    check("what might happen produces fear",
          list(dark) == ["fear"], dark)

    a = Affect()
    for _ in range(200):
        a.feel(Appraisal(valence=-0.6, anticipated=True, certainty=0.6,
                         control=0.2).emotions(), scale=0.45)
        for _ in range(4):
            a.decay()
    check("a long run of bad hours becomes chronic stress", a.stress > 0.5,
          f"{a.stress:.2f}")
    check("stress suppresses the will to act", a.drive() < 0.8, f"{a.drive():.2f}")


def test_memory():
    print("memory")
    m = Memory()
    rob = Appraisal(valence=-0.9, agency="other", control=0.2, norm=-1.0).emotions()
    m.encode(Episode(tick=100, kind="robbed", text="robbed on Crenshaw",
                     district="Leimert Park", people=("Vernon",), emotions=rob))
    for t in range(6):
        m.encode(Episode(tick=200 + t, kind="ate", text="lunch",
                         district="Downtown", place="philippes",
                         emotions={"joy": 0.1}))
    now = 6000
    bad, dull = m.episodes[0], m.episodes[-1]
    check("what you felt is what you keep", bad.salience > 3 * dull.salience,
          f"{bad.salience:.2f} vs {dull.salience:.2f}")
    check("the worst of it stays reachable while the rest fades",
          bad.retrievability(now) > 0.4 > dull.retrievability(now),
          f"{bad.retrievability(now):.2f} vs {dull.retrievability(now):.2f}")
    check("recall is cued by where you are",
          m.recall(now, district="Leimert Park")[0][1] is bad)
    check("recall is cued by who is there",
          m.recall(now, people=("Vernon",))[0][1] is bad)
    check("a place where it happened is dreaded",
          m.dread(now, district="Leimert Park") > m.dread(now, district="Downtown"))
    before = bad.salience
    for _ in range(50):
        m.dread(now, district="Leimert Park")
    check("weighing options does not rehearse memory",
          bad.salience == before, f"{before:.3f} -> {bad.salience:.3f}")
    check("repeated harm becomes a belief about the district",
          m.danger_of("Leimert Park") > m.danger_of("Downtown"))


def test_social_and_self():
    print("social and self")
    s = Social()
    for t in range(15):
        s.met("Ruth Okada", t * 10, quality=0.9)
    s.treated("Ruth Okada", 200, valence=0.6)
    s.treated("Amos Pym", 300, valence=-0.7, betrayal=0.6)
    check("trust follows how you were treated",
          s.of("Ruth Okada").trust > s.of("Amos Pym").trust)
    check("betrayal leaves a grudge", s.of("Amos Pym").grudge > 0.3)
    check("the closest person is the one you know and like",
          s.closest(1)[0].name == "Ruth Okada")
    check("you stay wary of the one who wronged you",
          "Amos Pym" in [r.name for r in s.feared()])

    m = SelfModel()
    for _ in range(60):
        m.did("work", True)
    for _ in range(60):
        m.did("provision", False)
    check("competence follows evidence",
          m.can("work") > 0.8 > 0.2 > m.can("provision"),
          f"work {m.can('work'):.2f} provision {m.can('provision'):.2f}")
    m.expectation["payday"] = 0.2
    surprise = m.outcome("payday", 0.9)
    check("a better outcome than expected is a positive surprise",
          surprise > 0.4 and m.surprises == 1, f"{surprise:.2f}")
    check("surprise moves what the unit expects next time",
          m.expect("payday") > 0.2)


def test_deliberation():
    print("deliberation")
    s = Simulation(n_units=6, seed=11)
    s.clock.tick = 11 * world.TICKS_PER_HOUR
    u = s.units[0]
    u.employed = True
    u.location_key = u.home_key
    u.district = u.home.district

    opts = generate(s, u)
    check("options are concrete places, not bare intents",
          any(o.venue for o in opts) and len(opts) > len(INTENTS))
    deliberate(s, u)
    kinds = [o.intent for o in u.considered]
    check("only the best option of each kind is weighed",
          len(kinds) == len(set(kinds)), kinds)
    check("every option records why it scored as it did",
          all(o.reasons for o in u.considered))

    # a unit with nothing scores work above a night out; a comfortable one does not
    def score_of(unit, intent):
        return next((o.score for o in unit.considered if o.intent == intent), None)
    u.funds, u.debt, u.hunger = 0.0, 5 * world.DAILY_COST, 0.2
    u.affect = Affect()
    deliberate(s, u)
    broke_gap = score_of(u, "work") - (score_of(u, "socialize") or -9)
    u.funds, u.debt = 40 * world.DAILY_COST, 0.0
    deliberate(s, u)
    rich_gap = score_of(u, "work") - (score_of(u, "socialize") or -9)
    check("need for money pulls a unit towards work", broke_gap > rich_gap,
          f"broke {broke_gap:+.2f} vs comfortable {rich_gap:+.2f}")

    # somewhere it was hurt scores worse than somewhere it was not
    v = next(x for x in world.VENUES if x.kind == "social" and x.district != u.district)
    from .cognition import Option
    plain = evaluate(s, u, Option("socialize", v.key))
    u.memory.encode(Episode(tick=s.clock.tick - 10, kind="robbed",
                            text="robbed", district=v.district, place=v.key,
                            emotions=Appraisal(valence=-0.9, agency="other",
                                               control=0.2, norm=-1.0).emotions()))
    after = evaluate(s, u, Option("socialize", v.key))
    check("a unit avoids where it was hurt", after < plain - 0.3,
          f"{plain:+.2f} -> {after:+.2f}")


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


def test_purchases_happen_once():
    """An hour of 'eat' is one meal, not one per tick."""
    print("commitments")
    s = Simulation(n_units=1, seed=4)
    u = s.units[0]
    diner = next(v for v in world.VENUES if v.kind == "food" and v.open_at(12))
    u.location_key = diner.key
    u.district = diner.district
    u.funds = 500.0
    u.hunger = 1.0
    s.clock.tick = 12 * world.TICKS_PER_HOUR
    crowd = s._crowd_by_venue()
    before = u.funds
    u.intent, u.target = "eat", diner.key
    s._perform(u, crowd, first=True)
    for _ in range(3):
        s._perform(u, crowd, first=False)
    check("an hour at a counter buys one meal",
          abs((before - u.funds) - diner.cost) < 1e-9,
          f"spent {before - u.funds:.2f} on a {diner.cost:.2f} meal")


def _fire_once():
    """Dismiss one unit and hand back the episode it laid down."""
    t = Simulation(n_units=1, seed=77)
    v = t.units[0]
    t.clock.tick = 7 * world.TICKS_PER_DAY
    while v.employed:
        t.units[0].week_ticks = 0
        t._daily(v)
    return next(e for e in v.memory.episodes if e.kind == "lost_job")


def test_worldview():
    """Units believe in an Earth that was never built, and can find the seam."""
    print("worldview")
    w = Worldview()
    check("the account starts settled", w.settled() and w.confidence == 1.0)
    check("a unit believes in places that are not simulated",
          w.believes_in("New York") and "New York" not in world.DISTRICTS)

    one = Worldview()
    for _ in range(40):
        one.probe("edge")
    many = Worldview()
    for seam in ("edge", "sky", "horizon", "time"):
        many.probe(seam)
    check("worrying at one seam habituates", one.confidence > 0.5,
          f"40 trips to the water leaves {one.confidence:.2f}")
    check("independent seams are what actually erode the account",
          many.confidence < one.confidence,
          f"four seams {many.confidence:.2f} vs forty trips {one.confidence:.2f}")

    child = inherit(many)
    check("a child inherits its parent's doubt, softened",
          many.confidence < child.confidence < 1.0, f"{child.confidence:.2f}")

    s = Simulation(n_units=12, seed=2010)
    s.run_days(150)
    shaken = min(s.units, key=lambda u: u.worldview.confidence)
    check("units out in the world find the edges of it",
          shaken.worldview.contradictions > 0)
    check("dissonance is derived from the account, not counted",
          abs(shaken.dissonance - 2 * (1 - shaken.worldview.confidence)) < 1e-9)
    check("what a shaken unit remembers most is the seam",
          any(e.kind.startswith("seam_")
              for e in sorted(shaken.memory.episodes,
                              key=lambda e: -e.salience)[:4]))


def test_mortality():
    """Nothing meant anything until units could stop existing."""
    print("mortality")
    u = Unit.spawn(np.random.default_rng(2))
    u.hunger, u.health = 0.95, 0.3
    high = u.peril()
    u.hunger, u.health = 0.2, 1.0
    check("peril tracks how close the end is", high > 0.5 > u.peril(),
          f"{high:.2f} vs {u.peril():.2f}")

    # starve one unit deliberately and watch the city notice
    s = Simulation(n_units=8, seed=21)
    s.run_days(20)
    victim = s.units[0]
    witness = max(s.units[1:], key=lambda o: o.social.of(victim.name).familiarity)
    witness.social.met(victim.name, s.clock.tick, quality=1.0)
    for _ in range(8):
        witness.social.met(victim.name, s.clock.tick, quality=1.0)
    before = len(s.units)
    victim.health, victim.hunger = 0.0, 0.95
    s._reap()
    check("a unit with no health left dies", len(s.units) == before - 1)
    check("the death is on the record",
          s.dead and s.dead[-1][2] == "starvation", s.dead[-1][2] if s.dead else None)
    grief = [e for e in witness.memory.episodes if e.kind == "bereaved"]
    check("the people who knew them grieve", len(grief) == 1)
    check("grief is the most memorable thing that has happened to them",
          grief and grief[0].salience >= max(e.salience
                                             for e in witness.memory.episodes))
    check("grief is sadness, not fear or anger",
          grief and max(grief[0].emotions, key=grief[0].emotions.get) == "sadness",
          grief[0].emotions if grief else None)

    m = SelfModel()
    start = m.can("coping")
    for _ in range(30):
        m.survived(0.6, 0.2)
    check("surviving a bad stretch is evidence a unit can cope",
          m.can("coping") > start and m.close_calls == 30,
          f"{start:.2f} -> {m.can('coping'):.2f}")
    check("coping is no longer frozen for every unit alike",
          m.can("coping") != 0.5)

    thinking = Simulation(n_units=14, seed=5)
    thinking.run_days(120)
    flailing = Simulation(n_units=14, seed=5, deliberate=False)
    flailing.run_days(120)
    check("thinking about it is what keeps units alive",
          len(thinking.dead) < len(flailing.dead),
          f"{len(thinking.dead)} dead vs {len(flailing.dead)} choosing at random")


def test_struggle():
    """The grind has to actually bite, or none of the feeling means anything."""
    print("struggle")
    s = Simulation(n_units=14, seed=6)
    s.run_days(120)
    hardship = [u for u in s.units if u.debt > 0 or u.affect.stress > 0.3]
    felt = [u for u in s.units if any(e.salience > 0.5 for e in u.memory.episodes)]

    # Dismissal is rare by design, so test the mechanism rather than hope the
    # population happened to show one in the window.
    fired = 0
    for trial in range(40):
        t = Simulation(n_units=1, seed=500 + trial)
        v = t.units[0]
        t.clock.tick = 7 * world.TICKS_PER_DAY
        t.units[0].week_ticks = 0                      # a week with no work at all
        t._daily(v)
        fired += not v.employed
    check("staying away from work costs a unit the job", 5 < fired < 40, fired)
    check("a unit who loses the job feels it",
          any(e.kind == "lost_job" and e.salience > 0.5
              for e in Simulation(n_units=1, seed=501).units[0].memory.episodes
              or [_fire_once()]))
    check("hardship shows up as debt or stress", len(hardship) >= 3, len(hardship))
    check("units are carrying memories that mattered", len(felt) >= 7, len(felt))
    check("relationships form between units",
          max(u.social.known() for u in s.units) >= 3)
    check("nobody is stuck feeling nothing",
          any(u.affect.dominant()[0] != "settled" for u in s.units))


def test_units_learn():
    """The load-bearing claim: they are self-learning, not scripted."""
    print("learning")
    thinking = Simulation(n_units=14, seed=3)
    thinking.run_days(140)
    flailing = Simulation(n_units=14, seed=3, deliberate=False)
    flailing.run_days(140)
    a, b = thinking.stats(), flailing.stats()
    check("deliberation keeps units fed", a["hunger"] < b["hunger"] - 0.05,
          f"{a['hunger']:.2f} vs {b['hunger']:.2f} choosing at random")
    check("deliberation keeps units solvent", a["debt"] < b["debt"],
          f"${a['debt']:.0f} vs ${b['debt']:.0f} choosing at random")
    check("deliberation keeps units in work", a["unemployed"] <= b["unemployed"],
          f"{a['unemployed']} vs {b['unemployed']} choosing at random")
    check("deliberation leaves units better off overall",
          a["wellbeing"] > b["wellbeing"],
          f"{a['wellbeing']:.3f} vs {b['wellbeing']:.3f} choosing at random")
    check("units have formed beliefs about the city",
          sum(len(u.memory.places) for u in thinking.units) / len(thinking.units) >= 3)

    worked_weekday, worked_weekend = 0, 0
    for _ in range(7):
        counter = [0]
        workday = thinking.clock.is_workday
        thinking.run_days(1, on_tick=lambda sim: counter.__setitem__(
            0, counter[0] + sum(1 for u in sim.units if u.worked_last)))
        if workday:
            worked_weekday += counter[0]
        else:
            worked_weekend += counter[0]
    check("units learned the working week",
          worked_weekday > 0 and worked_weekend == 0,
          f"weekday={worked_weekday} weekend={worked_weekend}")

    night = day = 0
    for _ in range(world.TICKS_PER_DAY * 3):
        thinking.step()
        asleep = sum(1 for u in thinking.units if u.activity == "asleep")
        if thinking.clock.hour < 6 or thinking.clock.hour >= 22:
            night += asleep
        elif 10 <= thinking.clock.hour < 18:
            day += asleep
    check("units sleep at night rather than in the afternoon",
          night > day * 1.3, f"night={night} day={day}")


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
    check("look() describes somewhere and when", world.EPOCH in link.look())
    msg = link.jack_out()
    check("link releases the body", not u.linked and "SEVERED" in msg)
    check("the unit noticed the missing hours", u.dissonance > 0,
          f"{u.dissonance:.3f}")
    check("the gap is in its memory",
          any(m.kind == "seam_time" for m in u.memory.episodes))
    check("missing time costs the unit its account of the world",
          u.worldview.confidence < 1.0, f"{u.worldview.confidence:.2f}")

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
    for t in (test_world, test_learner_finds_reward, test_circadian, test_unit,
              test_sim_runs_unattended, test_purchases_happen_once,
              test_appraisal, test_memory, test_social_and_self,
              test_deliberation, test_worldview, test_mortality,
              test_struggle, test_units_learn, test_link,
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
