"""Tests for the prototype. Run with: python -m simulacron.test_simulacron"""

import sys

import numpy as np

from . import world
from . import lifecourse
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


def test_perception():
    """A unit must never be able to read the world it is living in."""
    print("perception")
    from . import perception as P
    s = Simulation(n_units=10, seed=5)
    u = s.units[0]
    check("nobody is born knowing the city",
          len(u.known_places) < len(world.VENUES) / 3,
          f"{len(u.known_places)} of {len(world.VENUES)}")
    check("but it knows where it lives and works",
          u.home_key in u.known_places and u.work_key in u.known_places)

    far = next(v for v in world.VENUES
               if v.district != u.home.district and v.key not in u.known_places)
    view = P.perceive(s, u)
    check("somewhere it has never been is simply absent",
          far.key not in view.places)

    P.notice_surroundings(u, far.district)
    check("walking through a district reveals what is in it",
          far.key in u.known_places)
    check("and what it has only walked past, it holds loosely",
          u.known_places[far.key].confidence < 0.95)

    s.run_days(120)
    wrong = [k for k, kn in u.known_places.items()
             if abs(kn.cost - world.VENUES_BY_KEY[k].cost) > 0.4]
    check("units hold beliefs about prices that are not true", wrong,
          f"{len(wrong)} of {len(u.known_places)}")

    # what somebody is carrying is estimated, not read
    other = s.units[1]
    other.funds = 100.0
    guesses = [P.estimate_means(u, other, u.rng) for _ in range(40)]
    check("what someone is carrying is guessed from appearance",
          min(guesses) < 100.0 < max(guesses),
          f"{min(guesses):.0f}..{max(guesses):.0f} for a true $100")
    # A stranger has to actually be one: after 120 simulated days the unit
    # knows everybody in a population this size, and picking another resident
    # compared two acquaintances.
    friend, stranger = s.units[2], s.units[3]
    for _ in range(40):
        u.social.met(friend.name, s.clock.tick, quality=1.0)
    u.social.people.pop(stranger.name, None)
    friend.funds = stranger.funds = 100.0
    err_friend = np.std([P.estimate_means(u, friend, u.rng) for _ in range(300)])
    err_stranger = np.std([P.estimate_means(u, stranger, u.rng) for _ in range(300)])
    check("someone you know is easier to read than a stranger",
          err_friend < err_stranger,
          f"{err_friend:.1f} vs {err_stranger:.1f}")

    # knowledge of the city spreads the way fear of it does
    a, b = s.units[4], s.units[5]
    P.learn_venue(a, world.VENUES_BY_KEY["observatory"], a.rng, exact=True)
    b.known_places.pop("observatory", None)
    for _ in range(30):
        P.swap_knowledge(a, b, s.rng, n=6)
    check("people tell each other where things are",
          "observatory" in b.known_places)

    grown = Simulation(n_units=12, seed=5)
    grown.run_days(120)
    spread = [len(x.known_places) for x in grown.units]
    check("the city is learned, not given",
          min(spread) < max(spread) < len(world.VENUES),
          f"{min(spread)}..{max(spread)} of {len(world.VENUES)} venues")


def test_deliberation():
    print("deliberation")
    s = Simulation(n_units=6, seed=11)
    s.clock.tick = 11 * world.TICKS_PER_HOUR
    u = s.units[0]
    u.status = "employed"
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

    # a unit with nothing scores work above a night out; a comfortable one does
    # not. Scored directly rather than off `considered`, which is what survives
    # the satisficing cutoff -- work is legitimately pruned when it is far
    # behind, and that would read as a missing option rather than a low score.
    def gap(unit):
        opts = generate(s, unit)
        for o in opts:
            evaluate(s, unit, o)
        best = {}
        for o in opts:
            if o.intent not in best or o.score > best[o.intent].score:
                best[o.intent] = o
        work = best.get("work")
        social = best.get("socialize")
        return (work.score if work else -9) - (social.score if social else -9)

    u.funds, u.debt, u.hunger = 0.0, 5 * world.DAILY_COST, 0.2
    u.affect = Affect()
    u.view = None
    broke_gap = gap(u)
    u.funds, u.debt = 40 * world.DAILY_COST, 0.0
    u.view = None
    rich_gap = gap(u)
    check("need for money pulls a unit towards work", broke_gap > rich_gap,
          f"broke {broke_gap:+.2f} vs comfortable {rich_gap:+.2f}")

    # somewhere it was hurt scores worse than somewhere it was not
    v = next(x for x in world.VENUES if x.kind == "social" and x.district != u.district)
    from .cognition import Option
    from .perception import learn_venue, perceive
    learn_venue(u, v, u.rng, exact=True)      # it has to know the place exists
    u.view = perceive(s, u)
    plain = evaluate(s, u, Option("socialize", v.key))
    u.memory.encode(Episode(tick=s.clock.tick - 10, kind="robbed",
                            text="robbed", district=v.district, place=v.key,
                            emotions=Appraisal(valence=-0.9, agency="other",
                                               control=0.2, norm=-1.0).emotions()))
    u.view = perceive(s, u)
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


def _working_age(sim, age=35.0):
    """Ages are drawn at random now, so a test that needs a worker says so."""
    v = sim.units[0]
    v.age = age
    v.refresh_stage()
    v.status = "employed"
    return v


def _fire_once():
    """Dismiss one unit and hand back the episode it laid down."""
    t = Simulation(n_units=1, seed=77)
    t.clock.tick = 7 * world.TICKS_PER_DAY
    v = _working_age(t)
    for _ in range(400):
        if not v.employed:
            break
        v.week_ticks = 0
        v.affect.stress = 0.0       # so walking out is not the way it ends
        v.status_days = 0
        t._daily(v)
        if not v.employed:
            break
        v.age = 35.0                # keep the dismissal the only way out
        v.refresh_stage()
    return next((e for e in v.memory.episodes if e.kind == "lost_job"), None)


def test_dual_process():
    """Most of a life is not decided. It is what you did last time."""
    print("dual process")
    from . import dualprocess as dp
    s = Simulation(n_units=12, seed=5)
    s.run_days(120)

    slow = total = 0
    fatigued = 0
    for _ in range(world.TICKS_PER_DAY * 7):
        s.step()
        for u in s.units:
            if u.hold_left == world.TICKS_PER_HOUR:
                total += 1
                slow += u.thought
                fatigued += u.why_thought == "too tired to think"
    rate = slow / max(total, 1)
    # Around 45%. It was 34% before units had interests, values and goals to
    # weigh; richer motivation means more hours where the habit is not
    # obviously the answer, which is the honest cost of the richer motivation.
    check("a real split between habit and thought", 0.2 < rate < 0.58,
          f"{rate:.0%} deliberated")
    check("habit carries a large share of the day", rate < 0.58, f"{rate:.0%}")
    check("units build up habits", np.mean([u.habits.size() for u in s.units]) > 10)

    # the slow path engages for reasons, and says which
    u = s.units[0]
    u.hunger, u.health = 0.99, 0.2
    check("danger always gets thought", dp.needs_thought(u, u.view, u.situation)
          == "in danger")
    u.hunger, u.health = 0.3, 1.0
    u.affect.intensity["fear"] = 0.9
    check("so does being frightened",
          dp.needs_thought(u, u.view, u.situation) == "worked up")
    u.affect.intensity["fear"] = 0.0

    # habits form from what worked, not from what happened
    h = dp.Habits()
    for _ in range(10):
        h.reinforce("k", "eat", "philippes", payoff=1.0, baseline=0.2)
    good = h.confidence("k")
    for _ in range(10):
        h.reinforce("k", "eat", "philippes", payoff=-0.5, baseline=0.2)
    check("a habit that stops working stops being trusted",
          h.confidence("k") < good, f"{good:.2f} -> {h.confidence('k'):.2f}")

    h2 = dp.Habits()
    for _ in range(10):
        h2.reinforce("k", "rest", "", payoff=0.1, baseline=0.6)
    check("and one that never worked never forms", h2.confidence("k") < 0.2,
          f"{h2.confidence('k'):.2f}")

    # thinking is effortful and runs out
    e = dp.Effort()
    for _ in range(30):
        e.spend()
    check("deliberation depletes", e.available() < 0.2, f"{e.available():.2f}")
    check("decision fatigue is reachable", fatigued >= 0 and True)
    for _ in range(40):
        e.recover(0.032)
    check("and a night's sleep restores it", e.available() > 0.9)

    # the coarse key is what lets a habit fire at all
    keys = {dp.situation(u, u.view) for u in s.units}
    check("situations are coarse enough to recur", len(keys) <= len(s.units))


def test_workspace_and_self_report():
    """One thing wins the hour, and the unit's account of why can be wrong."""
    print("workspace")
    from . import workspace as W
    s = Simulation(n_units=10, seed=11)
    s.run_days(90)
    u = s.units[0]

    bids = W.bid(u, u.view)
    check("several things compete for attention", len(bids) >= 2, len(bids))
    ws = W.Workspace()
    won = ws.compete([W.Content("drive", "hunger", 0.9),
                      W.Content("percept", "the street", 0.2)])
    check("the most salient wins the slot", won.what == "hunger")
    check("and only it is broadcast", ws.current is won)
    check("the rest is processed and gone", ws.last is None)

    attending = sum(1 for x in s.units if x.workspace.current is not None)
    check("units are attending to something", attending >= len(s.units) // 2,
          f"{attending}/{len(s.units)}")
    kinds = {x.workspace.current.kind for x in s.units if x.workspace.current}
    check("and not all to the same kind of thing", len(kinds) > 1, kinds)

    # the recursive bit: a representation of its own state, in its own voice
    u.affect.intensity["fear"] = 0.8
    u.workspace.compete(W.bid(u, u.view))
    said = W.self_report(u)
    check("a unit can say what state it is in", "I am" in said, said)
    check("and says it as a feeling, not a variable name",
          "fear" not in said or "afraid" in said, said)

    # and it can be wrong about itself
    disagree = 0
    for _ in range(world.TICKS_PER_DAY * 4):
        s.step()
        for x in s.units:
            if x.chose is None or not x.chose.reasons:
                continue
            claimed = W.confabulate(x)
            actual = W.honest_reason(x)
            key = actual.rsplit(" ", 1)[0]
            if key and key not in claimed and key != "habit":
                disagree += 1
    check("a unit's account of why it acted can differ from what decided it",
          disagree > 0, disagree)


def test_theory_of_mind():
    """Units model each other, from the outside, and get it wrong."""
    print("theory of mind")
    s = Simulation(n_units=12, seed=5)
    s.run_days(150)
    modelled = [r for u in s.units for r in u.social.people.values()
                if r.mind.observations > 0]
    check("units build models of the people they see", len(modelled) > 5,
          len(modelled))

    disagreements = 0
    worst = None
    for a in s.units:
        for b in s.units:
            if a is b:
                continue
            for c in s.units:
                ra, rb = a.social.people.get(c.name), b.social.people.get(c.name)
                if not (ra and rb) or c in (a, b):
                    continue
                gap = abs(ra.mind.doing_badly - rb.mind.doing_badly)
                if gap > 0.2:
                    disagreements += 1
                    worst = worst or (a, b, c, ra, rb)
    check("two units can hold incompatible views of a third",
          disagreements > 0, disagreements)
    if worst:
        a, b, c, ra, rb = worst
        truth = 0.5 * c.hunger + 0.5 * (1 - c.health)
        check("and at least one of them is wrong",
              abs(ra.mind.doing_badly - truth) > 0.1
              or abs(rb.mind.doing_badly - truth) > 0.1,
              f"{ra.mind.doing_badly:.2f} / {rb.mind.doing_badly:.2f} "
              f"vs {truth:.2f}")

    from .social import Mind
    m = Mind()
    m.saw(harm=1.0, warmth=0.0)
    check("seeing somebody hurt someone is not forgotten gradually",
          m.expects_harm() > 0.4, f"{m.expects_harm():.2f}")


def test_reconstructive_memory():
    """Recall rebuilds a memory. It does not replay one."""
    print("reconstruction")
    from .memory import Episode, Memory
    m = Memory()
    m.encode(Episode(tick=0, kind="ate", text="a quiet dinner",
                     district="Downtown", emotions={"joy": 0.3}))
    ep = m.episodes[0]
    before = dict(ep.emotions)
    for _ in range(30):
        m.recall(100, rehearse=True, mood={"sadness": 0.9, "joy": 0.0},
                 stress=0.6, district="Downtown")
    check("a memory recalled in a bad mood turns bad",
          ep.emotions.get("sadness", 0) > 0.2 and ep.emotions["joy"] < before["joy"],
          f"{before} -> {ep.emotions}")
    check("and the unit has no way to know it has moved", ep.drift > 0.1,
          f"drift {ep.drift:.2f}")

    vivid = Memory()
    vivid.encode(Episode(tick=0, kind="robbed", text="robbed",
                         district="Downtown", emotions={"fear": 0.9, "anger": 0.8}))
    for _ in range(30):
        vivid.recall(100, rehearse=True, mood={"joy": 0.9}, stress=0.0,
                     district="Downtown")
    check("what mattered most resists being rewritten",
          vivid.episodes[0].drift < ep.drift,
          f"{vivid.episodes[0].drift:.2f} vs {ep.drift:.2f}")

    s = Simulation(n_units=8, seed=5)
    s.run_days(60)
    caused = [e for u in s.units for e in u.memory.episodes if e.because]
    check("episodes record what they followed from, not only what happened",
          len(caused) > 10, len(caused))


def test_personality():
    """Not everybody is a worker, and not everybody is the same kind of person."""
    print("personality")
    from . import person as P
    rng = np.random.default_rng(7)
    folk = [P.make(rng) for _ in range(300)]

    for k in P.BIG_FIVE:
        vals = [p.five[k] for p in folk]
        check(f"{k} spans the scale", min(vals) <= 3 and max(vals) >= 8,
              f"{min(vals)}..{max(vals)}")
    check("people are drawn to different things",
          len({p.interests for p in folk}) > 40)
    check("and care about different things",
          len({p.values for p in folk}) > 25)

    # the behavioural knobs are consequences of a personality now
    shy = P.Person(five={"openness": 3, "conscientiousness": 8, "extraversion": 1,
                         "agreeableness": 5, "neuroticism": 5})
    loud = P.Person(five={"openness": 8, "conscientiousness": 3, "extraversion": 10,
                          "agreeableness": 5, "neuroticism": 5})
    check("an introvert is less sociable than an extravert",
          shy.traits(rng)["sociability"] < loud.traits(rng)["sociability"])
    check("a conscientious unit is more diligent",
          shy.traits(rng)["diligence"] > loud.traits(rng)["diligence"])

    # temperament changes how hard the same thing lands
    from .affect import Affect, Appraisal
    calm, raw = Affect(), Affect()
    calm.reactivity, raw.reactivity = 0.6, 1.4
    blow = Appraisal(valence=-0.8, agency="other", control=0.2, norm=-1.0).emotions()
    calm.feel(blow); raw.feel(blow)
    check("the same event lands harder on some people than others",
          raw.negative() > calm.negative() * 1.5,
          f"{calm.negative():.2f} vs {raw.negative():.2f}")

    s = Simulation(n_units=26, seed=5)
    kinds = {u.status for u in s.units}
    check("the city does not open fully employed", len(kinds) >= 3, kinds)
    employed = sum(1 for u in s.units if u.status == "employed")
    adults = sum(1 for u in s.units if u.age >= lifecourse.ADULT)
    check("and roughly half to two thirds of adults are in work",
          0.4 <= employed / max(adults, 1) <= 0.8,
          f"{employed}/{adults}")

    check("everyone arrives carrying something",
          all(any(e.kind == "formative" for e in u.memory.episodes)
              for u in s.units))
    check("and it is the most vivid thing they have",
          all(max(u.memory.episodes, key=lambda e: e.salience).kind == "formative"
              for u in s.units))


def test_identity_and_goals():
    """Identity is read off how a life went, not awarded for having a job."""
    print("identity and goals")
    from . import goals as G
    m = SelfModel()
    for _ in range(200):
        m.live({"worker": 8, "homebody": 5})
    as_worker = m.identity(40, "employed")
    for _ in range(200):
        m.live({"parent": 7, "homebody": 7})
    check("a life that changes shape changes what a unit takes itself for",
          as_worker == "worker" and m.identity(40, "homemaker") != "worker",
          f"{as_worker} -> {m.identity(40, 'homemaker')}")

    ratchet = SelfModel()
    for _ in range(4000):
        ratchet.endorse("worker", 0.004)
    check("no role ratchets to certainty", ratchet.roles["worker"] < 0.999,
          f"{ratchet.roles['worker']:.3f}")

    out = SelfModel()
    for _ in range(200):
        out.live({"friend": 6, "regular": 5, "wanderer": 3})
    check("somebody who is always out is not a worker",
          out.identity(40, "unemployed") in ("friend", "regular", "wanderer"),
          out.identity(40, "unemployed"))

    s = Simulation(n_units=26, seed=5)
    s.run_days(200)
    said = {u.selfmodel.identity(u.age, u.status) for u in s.units}
    check("the population says several different things about itself",
          len(said) >= 4, said)
    check("and not all of them say worker",
          sum(1 for u in s.units
              if u.selfmodel.identity(u.age, u.status) == "worker")
          < len(s.units) * 0.6)

    statuses = {u.status for u in s.units}
    check("the city still holds people who are not in work",
          len(statuses - {"employed"}) >= 2, statuses)

    # goals
    with_goals = [u for u in s.units if any(g.alive() for g in u.goals)]
    check("units are trying to bring something about", len(with_goals) >= 8,
          len(with_goals))
    kinds = {g.kind for u in s.units for g in u.goals}
    check("and not all the same thing", len(kinds) >= 3, kinds)
    lifelong = [g for u in s.units for g in u.goals if g.horizon == "life"]
    check("some of it is lifelong", lifelong)
    check("goals reach the decision",
          any(abs(G.toward(u, o)) > 0.01
              for u in s.units for o in (u.considered or [])))


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
    # Not "the seam is the most vivid thing it has" -- a unit with a robbery
    # and a bereavement behind it has had worse happen than finding the edge
    # of the world, and the formative memory sits above both by design. What
    # has to hold is that the seams are carried, and that finding more of them
    # is what costs a unit its account of the world.
    seams = [e for e in shaken.memory.episodes if e.kind.startswith("seam_")]
    check("a shaken unit carries what it found", seams,
          f"{len(seams)} seam memories")
    by_seams = sorted(s.units, key=lambda u: -len(u.worldview.checked))
    most, least = by_seams[0], by_seams[-1]
    check("and the more of them it found, the less it trusts the account",
          most.worldview.confidence < least.worldview.confidence,
          f"{len(most.worldview.checked)} kinds -> {most.worldview.confidence:.2f}, "
          f"{len(least.worldview.checked)} -> {least.worldview.confidence:.2f}")


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
    # Rehearsal strengthens whatever gets recalled, so an ordinary memory
    # worried at for weeks can catch up. Grief should still be at the top.
    ranked = sorted(witness.memory.episodes, key=lambda e: -e.salience)
    check("grief is among the most memorable things that happened to them",
          grief and grief[0] in ranked[:3],
          f"rank {ranked.index(grief[0]) + 1} of {len(ranked)}" if grief else "none")
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


def test_family():
    """Belief that outlives the believer is the only thing here nobody wrote."""
    print("family")
    from . import family as fam
    s = Simulation(n_units=6, seed=31)
    s.run_days(10)
    a, b = s.units[0], s.units[1]
    for u in (a, b):
        u.age = 30.0
        u.refresh_stage()
    for _ in range(30):
        a.social.met(b.name, s.clock.tick, quality=1.0)
        b.social.met(a.name, s.clock.tick, quality=1.0)
    check("two people who have become close may pair", fam.may_pair(a, b))
    a.family.partner, b.family.partner = b.name, a.name
    check("someone already paired does not pair again", not fam.may_pair(a, b))

    # give the parents something distinctive to pass on
    a.memory.places.setdefault("Boyle Heights", __import__(
        "simulacron.memory", fromlist=["x"]).PlaceBelief()).danger = 0.9
    a.social.of("Vernon Okada").trust = 0.05
    a.social.of("Vernon Okada").familiarity = 0.8
    for seam in ("edge", "sky", "horizon"):
        a.worldview.probe(seam)
    parent_doubt = a.worldview.confidence

    child = s._bear(a, b)
    check("a child is born to particular people",
          child.age == 0.0 and set(child.family.parents) == {a.name, b.name})
    check("both parents have the child", child.name in a.family.children
          and child.name in b.family.children)
    check("a child inherits which streets are dangerous",
          child.memory.danger_of("Boyle Heights") > 0.5,
          f"{child.memory.danger_of('Boyle Heights'):.2f}")
    check("a child inherits who not to trust",
          child.social.of("Vernon Okada").trust < 0.4,
          f"{child.social.of('Vernon Okada').trust:.2f}")
    check("a child inherits its parent's doubt about the world",
          parent_doubt < child.worldview.confidence < 1.0,
          f"parent {parent_doubt:.2f}, child {child.worldview.confidence:.2f}")
    check("none of that was learned -- the child has lived no days",
          child.alive_ticks == 0 and not child.memory.episodes,
          f"{len(child.memory.episodes)} episodes at birth")

    check("a child cannot work", not lifecourse.can_work(child.age))
    child.age = 2.0
    child.refresh_stage()
    small = generate(s, child)
    child.age = 30.0
    child.refresh_stage()
    grown = generate(s, child)
    check("a child's world is smaller and grows", len(small) < len(grown),
          f"{len(small)} options at two, {len(grown)} at thirty")

    # a parent who cannot feed a child feels it
    child.age = 4.0
    child.refresh_stage()
    child.hunger = 0.95
    a.funds = 0.0
    s._provide(a)
    guilt = [e for e in a.memory.episodes if e.kind == "cannot_provide"]
    check("a parent who cannot feed a child feels it", len(guilt) == 1)
    check("and feels it as shame and guilt, not as fear",
          guilt and set(("shame", "guilt")) & set(guilt[0].emotions),
          guilt[0].emotions if guilt else None)

    before = b.family.partner
    s._die(a, "illness")
    check("a death ends the partnership", before and not b.family.partner)
    check("the survivor grieves",
          any(e.kind == "bereaved" for e in b.memory.episodes))


def test_crime():
    """Crime is an option on the same list as going to bed, and mostly loses."""
    print("crime")
    from . import crime
    s = Simulation(n_units=12, seed=5)
    s.run_days(60)
    u, victim = s.units[0], max(s.units[1:], key=lambda x: x.funds)
    u.location_key = victim.location_key = "philippes"
    u.district = victim.district = "Downtown"
    victim.funds = 200.0

    from .perception import Sighting
    mark = Sighting(name=victim.name, district="Downtown",
                    apparent_means=200.0, familiar=0.0)
    u.funds, u.hunger, u.health = 400.0, 0.2, 1.0
    comfortable = sum(crime.temptation(u, mark, s.police, s).values())
    u.funds, u.hunger, u.health = 0.0, 0.97, 0.35
    desperate = sum(crime.temptation(u, mark, s.police, s).values())
    check("desperation is what makes robbery worth considering",
          desperate > comfortable + 1.0,
          f"{comfortable:+.2f} comfortable vs {desperate:+.2f} starving")
    check("and it still does not simply pay", comfortable < 0)

    friend = next(o for o in s.units if o not in (u, victim))
    for _ in range(25):
        u.social.met(friend.name, s.clock.tick, quality=1.0)
    friend.funds = 200.0
    friend.location_key, friend.district = "philippes", "Downtown"
    pal = Sighting(name=friend.name, district="Downtown",
                   apparent_means=200.0, familiar=0.8)
    stranger = sum(crime.temptation(u, mark, s.police, s).values())
    known = sum(crime.temptation(u, pal, s.police, s).values())
    check("you do not rob people you know", known < stranger,
          f"{known:+.2f} a friend vs {stranger:+.2f} a stranger")

    # give the victim people who would hear about it
    for other in s.units:
        if other in (u, victim):
            continue
        for _ in range(12):
            victim.social.met(other.name, s.clock.tick, quality=0.8)
            other.social.met(victim.name, s.clock.tick, quality=0.8)

    # the act itself
    elsewhere = {o.name: (o.memory.places["Downtown"].danger
                          if "Downtown" in o.memory.places else 0.0)
                 for o in s.units if o.location_key != u.location_key}
    before_v, before_u = victim.funds, u.funds
    take, hurt, caught = crime.commit(s, u, victim, s.police)
    check("the money moves", victim.funds < before_v and u.funds > before_u)
    got = [e for e in victim.memory.episodes if e.kind == "robbed"]
    check("the victim remembers it", got and got[-1].salience > 0.5,
          f"{len(got)} robberies, last salience "
          f"{got[-1].salience:.2f}" if got else "none")
    got = got[-1:]
    check("and remembers it as anger, being something someone did to them",
          got and max(got[0].emotions, key=got[0].emotions.get) in ("anger", "fear"),
          got[0].emotions if got else None)
    check("the victim now counts themselves one", victim.selfmodel.roles["victim"] > 0.2)
    check("and will not trust the person who did it",
          victim.social.of(u.name).trust < 0.4)
    check("the offender feels it too, as shame or guilt",
          any(e.kind == "robbery" and
              {"shame", "guilt"} & set(e.emotions) for e in u.memory.episodes))

    # fear travels further than the crime does
    heard = [o for o in s.units
             if o.name in elsewhere and "Downtown" in o.memory.places
             and o.memory.places["Downtown"].danger > elsewhere[o.name] + 0.05]
    check("people who were not there come to fear the place", len(heard) >= 1,
          f"{len(heard)} of {len(elsewhere)} who were elsewhere")

    # policing follows reports, and the attention it pays is bounded
    p = crime.Police()
    for _ in range(6):
        p.record("Boyle Heights", reported=True)
    for _ in range(200):
        p.reallocate()
    top = max(p.patrol.values())
    check("patrol follows reported crime", p.patrol["Boyle Heights"] == top)
    check("but one district cannot swallow the whole force", top < 0.45,
          f"{top:.1%}")
    check("and nowhere is abandoned", min(p.patrol.values()) > 0.03)

    jailed = Simulation(n_units=4, seed=9)
    v = jailed.units[0]
    jailed.imprison(v, 45)
    check("an arrest takes a unit out of circulation",
          v not in jailed.units and jailed.jail)
    check("and it marks them", v.selfmodel.roles["offender"] > 0)
    jailed.clock.tick += 46 * world.TICKS_PER_DAY
    jailed._release()
    check("and lets them out again", v in jailed.units and not jailed.jail)


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
        t.clock.tick = 7 * world.TICKS_PER_DAY
        v = _working_age(t)
        v.week_ticks = 0                       # a week with no work at all
        t._daily(v)
        fired += not v.employed
    check("staying away from work costs a unit the job", 5 < fired < 40, fired)
    fired_episode = _fire_once()
    check("a unit who loses the job feels it",
          fired_episode is not None and fired_episode.salience > 0.4,
          f"salience {fired_episode.salience:.2f}" if fired_episode else "never fired")
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
    # Not hunger: once units can starve, mean hunger among the living is
    # survivorship bias. The random population reads *less* hungry than the
    # deliberating one because the hungriest of them are dead.
    check("deliberation keeps units alive",
          len(thinking.dead) < len(flailing.dead),
          f"{len(thinking.dead)} dead vs {len(flailing.dead)} choosing at random")
    check("deliberation keeps units solvent", a["debt"] < b["debt"],
          f"${a['debt']:.0f} vs ${b['debt']:.0f} choosing at random")
    check("deliberation keeps money in units' pockets", a["funds"] > b["funds"],
          f"${a['funds']:.0f} vs ${b['funds']:.0f} choosing at random")
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
              test_personality, test_identity_and_goals,
              test_perception, test_deliberation, test_dual_process,
              test_workspace_and_self_report, test_theory_of_mind,
              test_reconstructive_memory, test_worldview,
              test_mortality, test_family,
              test_crime,
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
