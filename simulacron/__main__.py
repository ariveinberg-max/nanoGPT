"""Console for the prototype.

    python -m simulacron run      --days 365      # let it run; nobody is watching
    python -m simulacron units    --days 365      # who is down there
    python -m simulacron rhythm   --days 365      # the day they taught themselves
    python -m simulacron jackin   --days 365      # walk around in 1937
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
    print(f"{'day':>5} {'reward':>8} {'wellbeing':>10} {'funds':>9} "
          f"{'hunger':>7} {'fatigue':>8} {'work h':>7} {'anom':>5}")
    for day in range(1, args.days + 1):
        worked = [0]
        sim.run_days(1, on_tick=lambda s: worked.__setitem__(
            0, worked[0] + sum(1 for u in s.units if u.worked_last)))
        if day % max(1, args.every) == 0 or day == args.days:
            st = sim.stats()
            print(f"{st['day']:>5} {st['reward']:>+8.3f} {st['wellbeing']:>10.3f} "
                  f"{st['funds']:>9.2f} {st['hunger']:>7.2f} {st['fatigue']:>8.2f} "
                  f"{worked[0] / args.units / world.TICKS_PER_HOUR:>7.1f} "
                  f"{st['anomalies']:>5}")
    for line in sim.log[-10:]:
        print(line)
    return sim


def cmd_units(args):
    sim = _build(args)
    for u in sorted(sim.units, key=lambda x: -x.funds)[:args.show]:
        print(u.describe())
        for m in u.recent_memories(3):
            print(f"  ...{m.text}")
        print()


def cmd_rhythm(args):
    """Sample the population's chosen intent hour by hour over a working week."""
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
                table[h][u.intent] = table[h].get(u.intent, 0) + 1
        days += 1
    print(f"The weekday {args.units} units taught themselves over {args.days} days")
    print(f"{'hour':>5}  {'dominant':<11} {'share':>6}  profile")
    for h in range(24):
        row = table[h]
        if not row:
            continue
        total = sum(row.values())
        top, n = max(row.items(), key=lambda kv: kv[1])
        bars = " ".join(f"{k[:4]}{'#' * max(1, round(9 * v / total))}"
                        for k, v in sorted(row.items(), key=lambda kv: -kv[1])[:3])
        print(f"{h:>5}  {top:<11} {n / total:>5.0%}  {bars}")


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
    p.add_argument("--seed", type=int, default=1937)
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

    j = sub.add_parser("jackin", help="link a mind into a unit")
    j.add_argument("--days", type=int, default=365)
    j.add_argument("--unit", default=None, help="name (or part of one)")
    j.set_defaults(fn=cmd_jackin)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
