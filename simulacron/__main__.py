"""Console for the prototype.

    python -m simulacron run      --days 365      # let it run; nobody is watching
    python -m simulacron units    --days 365      # who is down there
    python -m simulacron rhythm   --days 365      # the day they taught themselves
    python -m simulacron jackin   --days 365      # walk around in 2010
"""

import argparse
import sys

from . import world
from .brain import INTENTS
from .link import repl
from .sim import Simulation


def _build(args, announce=True):
    sim = Simulation(n_units=args.units, seed=args.seed)
    if args.days and announce:
        print(f"Bringing up the prototype: {args.units} units, "
              f"{args.days} days of unattended runtime...", file=sys.stderr)
    if args.days:
        sim.run_days(args.days)
    return sim


def cmd_run(args):
    sim = Simulation(n_units=args.units, seed=args.seed)
    print(f"SIMULACRON -- {world.EPOCH}, {args.units} units, no operator linked.")
    print(f"{'day':>5} {'wellbeing':>10} {'funds':>9} {'debt':>8} {'hunger':>7} "
          f"{'stress':>7} {'trauma':>7} {'jobless':>8} {'ill':>4} {'friends':>8} "
          f"{'work h':>7} {'anom':>5}")
    for day in range(1, args.days + 1):
        worked = [0]
        sim.run_days(1, on_tick=lambda s: worked.__setitem__(
            0, worked[0] + sum(1 for u in s.units if u.worked_last)))
        if day % max(1, args.every) == 0 or day == args.days:
            st = sim.stats()
            print(f"{st['day']:>5} {st['wellbeing']:>10.3f} {st['funds']:>9.2f} "
                  f"{st['debt']:>8.2f} {st['hunger']:>7.2f} {st['stress']:>7.2f} "
                  f"{st['trauma']:>7.3f} {st['unemployed']:>8} {st['ill']:>4} "
                  f"{st['friends']:>8.1f} "
                  f"{worked[0] / args.units / world.TICKS_PER_HOUR:>7.1f} "
                  f"{st['anomalies']:>5}")
    for line in sim.log[-12:]:
        print(line)
    return sim


def cmd_units(args):
    sim = _build(args)
    for u in sorted(sim.units, key=lambda x: -x.affect.stress)[:args.show]:
        print(u.describe())
        recent = u.memory.recent(3)
        if recent:
            print("  remembers:")
            for m in recent:
                print(f"    {m.text} (salience {m.salience:.2f})")
        print()


def cmd_mind(args):
    """One unit, in full: what it feels, believes, remembers and expects."""
    sim = _build(args)
    if args.unit:
        matches = [u for u in sim.units if args.unit.lower() in u.name.lower()]
        u = matches[0] if matches else None
    else:
        u = max(sim.units, key=lambda x: x.affect.stress + x.affect.trauma)
    if u is None:
        print("No such unit. Try: " + ", ".join(x.name for x in sim.units[:6]))
        return
    print(u.describe())
    print("\n  what it believes about places:")
    for line in u.memory.describe_places():
        print(f"    {line}")
    print("\n  who it knows:")
    for line in u.social.describe(6):
        print(f"    {line}")
    print("\n  what it carries:")
    for m in sorted(u.memory.episodes, key=lambda e: -e.salience)[:6]:
        felt = ", ".join(f"{k} {v:.2f}" for k, v in
                         sorted(m.emotions.items(), key=lambda kv: -kv[1])[:2])
        print(f"    [{m.salience:.2f}] {m.text}" + (f" -- {felt}" if felt else ""))
    print(f"\n  ruminated {u.rumination} times; "
          f"{u.selfmodel.surprises} surprises so far")
    if u.considered:
        print("\n  last time it decided:")
        for o in sorted(u.considered, key=lambda o: -o.score)[:5]:
            print(f"    {o.score:+.2f}  {o.label():40s} {o.why(3)}")


def cmd_rhythm(args):
    """What the population actually does, hour by hour, over a working week.

    Deliberately reports activity rather than intent: a unit that sets out for
    a shut workplace wanted to work but spent the hour idle, and only the
    second of those is a fact about the day it has learned.
    """
    sim = _build(args)
    table = {h: {} for h in range(24)}
    days = 0
    while days < 5:
        if not sim.clock.is_workday:
            sim.run_days(1)
            continue
        for _ in range(world.TICKS_PER_DAY):
            sim.step()
            h = sim.clock.hour
            for u in sim.units:
                table[h][u.activity] = table[h].get(u.activity, 0) + 1
        days += 1
    print(f"The weekday {args.units} units taught themselves over {args.days} days")
    print(f"{'hour':>5}  {'mostly':<11} {'share':>6}  profile")
    for h in range(24):
        row = table[h]
        if not row:
            continue
        total = sum(row.values())
        top, n = max(row.items(), key=lambda kv: kv[1])
        bars = " ".join(f"{k.split()[-1][:5]}{'#' * max(1, round(9 * v / total))}"
                        for k, v in sorted(row.items(), key=lambda kv: -kv[1])[:3])
        print(f"{h:>5}  {top:<11} {n / total:>5.0%}  {bars}")


def cmd_city(args):
    """The city as the lab sees it: offences, attention, and belief."""
    sim = _build(args)
    st = sim.stats()
    print(f"{world.EPOCH} + {st['day']} days")
    print(f"  {st['alive']} alive ({st['children']} children), {st['born']} born, "
          f"{st['dead']} dead, {st['jailed']} inside")
    print(f"  {st['robberies']} robberies, {st['unemployed']} out of work, "
          f"{st['anomalies']} asking questions about the shape of the world")
    print("\n  policing -- patrol follows reported crime, and nothing else:")
    for line in sim.police.describe():
        print(f"    {line}")
    print("\n  what the city believes about its own districts:")
    for d in world.DISTRICTS:
        held = [u.memory.places[d].danger for u in sim.units if d in u.memory.places]
        been = [u.memory.places[d].visits for u in sim.units if d in u.memory.places]
        if not held:
            continue
        secondhand = sum(1 for u in sim.units
                         if d in u.memory.places and u.memory.places[d].visits <= 3
                         and u.memory.places[d].danger > 0.2)
        print(f"    {d:14s} feared by {sum(1 for x in held if x > 0.2):2d}/"
              f"{len(sim.units):2d}  mean danger {sum(held)/len(held):.2f}"
              f"  ({secondhand} of them on hearsay)")
    if sim.dead:
        print("\n  the dead:")
        for tick, u, cause in sim.dead[-5:]:
            print(f"    {u.name}, {u.age:.0f}, of {cause}")


def cmd_serve(args):
    """Run the prototype and serve it. Ctrl-C to stop."""
    from .server import serve
    httpd, engine = serve(host=args.host, port=args.port, n_units=args.units,
                          seed=args.seed, warmup=args.warmup, hz=args.hz)
    where = f"http://{args.host}:{args.port}"
    print(f"\n  Simulacron is running at  {where}\n")
    print(f"  {args.units} units, seed {args.seed}. Warming up {args.warmup} days "
          f"before it opens.", file=sys.stderr)
    print("  Ctrl-C to stop.\n", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped. The prototype is gone.", file=sys.stderr)
        httpd.server_close()


def cmd_view(args):
    """Bake one run into a standalone page you can open in a browser."""
    from .record import build_page
    print(f"Recording {args.units} units over {args.days} days "
          f"(after {args.warmup} days of warm-up)...", file=sys.stderr)
    out = build_page(out=args.out, n_units=args.units, seed=args.seed,
                     days=args.days, warmup=args.warmup, every=args.every)
    import os
    print(f"{out}  ({os.path.getsize(out) / 1e6:.1f} MB) -- open it in a browser")


def cmd_jackin(args):
    sim = _build(args)
    if args.unit:
        matches = [u for u in sim.units if args.unit.lower() in u.name.lower()]
        if not matches:
            print(f"No unit named '{args.unit}'. Try: "
                  + ", ".join(u.name for u in sim.units[:6]))
            return
        unit = matches[0]
    else:
        unit = max(sim.units, key=lambda u: u.wellbeing())
    repl(sim, unit)


def main(argv=None):
    p = argparse.ArgumentParser(prog="simulacron", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--units", type=int, default=16, help="how many cyber beings")
    p.add_argument("--seed", type=int, default=2010)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the prototype unattended and report")
    r.add_argument("--days", type=int, default=365)
    r.add_argument("--every", type=int, default=30, help="report every N days")
    r.set_defaults(fn=cmd_run)

    u = sub.add_parser("units", help="describe the population")
    u.add_argument("--days", type=int, default=365)
    u.add_argument("--show", type=int, default=6)
    u.set_defaults(fn=cmd_units)

    y = sub.add_parser("rhythm", help="the daily rhythm the units learned")
    y.add_argument("--days", type=int, default=365)
    y.set_defaults(fn=cmd_rhythm)

    m = sub.add_parser("mind", help="one unit's inner life in full")
    m.add_argument("--days", type=int, default=365)
    m.add_argument("--unit", default=None, help="name (or part of one)")
    m.set_defaults(fn=cmd_mind)

    c = sub.add_parser("city", help="crime, policing and what people believe")
    c.add_argument("--days", type=int, default=365)
    c.set_defaults(fn=cmd_city)

    sv = sub.add_parser("serve", help="run the prototype and watch it live in a browser")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--warmup", type=int, default=90,
                    help="days to run before opening, so units know their way around")
    sv.add_argument("--hz", type=float, default=8.0,
                    help="simulated ticks per real second (4 ticks = 1 hour)")
    sv.set_defaults(fn=cmd_serve)

    v = sub.add_parser("view", help="record a run and build a playback console")
    v.add_argument("--days", type=int, default=150)
    v.add_argument("--warmup", type=int, default=90,
                   help="days to run before recording starts")
    v.add_argument("--every", type=int, default=4,
                   help="ticks between recorded frames (4 = hourly)")
    v.add_argument("--out", default="simulacron-console.html")
    v.set_defaults(fn=cmd_view)

    j = sub.add_parser("jackin", help="link a mind into a unit")
    j.add_argument("--days", type=int, default=365)
    j.add_argument("--unit", default=None, help="name (or part of one)")
    j.set_defaults(fn=cmd_jackin)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
