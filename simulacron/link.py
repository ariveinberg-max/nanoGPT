"""The link: an operator's mind, downstairs, walking around 2010.

While you are jacked in, your body stays in the lab and the unit you occupy
holds your consciousness. Its own policy is suspended -- it is not asleep, it
is simply not there. When you jack out it resumes mid-stride with a hole where
those hours should have been, and units notice holes.
"""

import textwrap

from . import world
from .brain import INTENTS

WRAP = textwrap.TextWrapper(width=76, subsequent_indent="  ")


class LinkSession:
    def __init__(self, sim, unit, quiet=False):
        self.sim = sim
        self.unit = unit
        self.quiet = quiet
        self.entered_tick = sim.clock.tick
        self.open = False

    # ------------------------------------------------------------- lifecycle
    def jack_in(self):
        u = self.unit
        if u.linked:
            raise RuntimeError(f"{u.name} is already carrying an operator.")
        u.linked = True
        u.pending_record = None
        u.hold_left = 0
        self.open = True
        self.entered_tick = self.sim.clock.tick
        return (f"LINK ESTABLISHED -- {u.name}\n"
                f"{world.EPOCH}. You are standing in "
                f"{u.location.name if u.location else 'the middle of a street'}"
                f", {u.district}.\n"
                f"Your body remains in the lab. Type 'help' for what this body knows.")

    def jack_out(self):
        u = self.unit
        u.linked = False
        self.open = False
        lost = (self.sim.clock.tick - self.entered_tick) / world.TICKS_PER_HOUR
        if lost >= 0.5:
            # Missing hours are a seam like any other: the unit queries its own
            # account of the day and finds nothing where the afternoon was.
            from . import cognition
            for _ in range(1 + int(lost // 3)):
                cognition.probe_world(self.sim, u, "time")
        return (f"LINK SEVERED after {lost:.1f} simulated hours.\n"
                f"{u.name} is walking again. Dissonance now "
                f"{u.dissonance:.2f}.")

    # ------------------------------------------------------------- the world
    def look(self):
        u = self.unit
        clock = self.sim.clock
        if u.travelling:
            return (f"{clock.stamp()} -- the {world.TRANSIT_NOUN}, somewhere between "
                    f"districts. {u.travel_left} stops to "
                    f"{world.VENUES_BY_KEY[u.travel_to].name}.")
        here = u.location
        others = [o for o in self.sim.units
                  if o is not u and o.location_key == u.location_key]
        lines = [f"{clock.stamp()}, {world.EPOCH} -- {here.name}, {u.district}."]
        if here.kind == "food":
            lines.append(f"A meal runs ${here.cost:.2f}. You have ${u.funds:.2f}.")
        elif here.kind == "social":
            lines.append(f"The door is ${here.cost:.2f}. You have ${u.funds:.2f}.")
        elif here.kind == "work":
            lines.append(f"Your shift pays ${here.wage:.2f} an hour.")
        if not here.open_at(clock.hour):
            lines.append("It is shut.")
        if others:
            lines.append("Also here: " + ", ".join(o.name for o in others) + ".")
        else:
            lines.append("Nobody you know.")
        lines.append("Reachable: " + ", ".join(
            sorted({v.name for v in world.venues_in(u.district)} - {here.name})
            or ["nothing else in this district"]))
        return "\n".join(WRAP.fill(l) for l in lines)

    def go(self, target):
        u = self.unit
        if u.travelling:
            return (f"Already aboard, {u.travel_left} stops from "
                    f"{world.VENUES_BY_KEY[u.travel_to].name}.")
        target = target.strip().lower()
        for v in world.VENUES:
            if target in v.name.lower() or target == v.key:
                self.sim._send(u, v.key)
                return f"Boarding for {v.name}. {u.travel_left} stops."
        for d in world.DISTRICTS:
            if target in d.lower():
                vs = world.venues_in(d)
                if vs:
                    self.sim._send(u, vs[0].key)
                    return f"Boarding for {d}. {u.travel_left} stops."
        return f"Nothing called '{target}' that you can get to."

    def do(self, intent):
        u = self.unit
        if u.travelling:
            return (f"You are in transit, {u.travel_left} stops from "
                    f"{world.VENUES_BY_KEY[u.travel_to].name}. Try 'wait'.")
        if intent not in INTENTS:
            return f"This body does not know how to '{intent}'."
        before = (u.funds, u.hunger, u.fatigue, u.loneliness)
        crowd = self.sim._crowd_by_venue()
        u.intent = intent
        u.target = u.location_key
        self.sim._perform(u, crowd, first=True)
        after = (u.funds, u.hunger, u.fatigue, u.loneliness)
        if u.travelling:
            return f"You set off to {intent}. {u.travel_left} stops."
        deltas = []
        for label, b, a in zip(("funds", "hunger", "fatigue", "loneliness"),
                               before, after):
            if abs(a - b) > 1e-6:
                deltas.append(f"{label} {b:+.2f}->{a:.2f}")
        return f"You {intent}. " + (", ".join(deltas) if deltas else "Nothing comes of it.")

    def wait(self, hours=1):
        for _ in range(int(hours * world.TICKS_PER_HOUR)):
            self.sim.step()          # step() already ages the linked body
        return f"{hours:g} hour(s) pass. " + self.look()

    def memories(self):
        m = self.unit.recent_memories(8)
        if not m:
            return "This body remembers nothing in particular."
        return "\n".join(f"  - {x.text}" for x in m)

    def status(self):
        return self.unit.describe()

    def inner(self):
        """What this body is carrying around: feeling, beliefs, people."""
        u = self.unit
        out = [f"  feeling   {u.affect.describe()}",
               f"  self      {u.selfmodel.narrative(u.affect)}"]
        if u.social.people:
            out.append("  people:")
            out += [f"    {line}" for line in u.social.describe(4)]
        places = u.memory.describe_places()
        if places:
            out.append("  places:")
            out += [f"    {line}" for line in places[:4]]
        return "\n".join(out)


HELP = """\
  look                 take in where this body is
  go <place|district>  board the Metro, or get in the car
  eat / work / sleep / socialize / rest / wander / errand / seek
  wait [hours]         let the prototype run around you
  who                  who else is in the district
  memories             what this body remembers
  inner                what it feels, believes and thinks of people
  status               drives, funds, dissonance
  jackout              return to your own body
"""


def repl(sim, unit, stream_in=None, stream_out=print):
    """Interactive link. Pass stream_in for scripted sessions."""
    session = LinkSession(sim, unit)
    stream_out(session.jack_in())
    stream_out(session.look())

    def _read():
        if stream_in is not None:
            return next(stream_in, "jackout")
        try:
            return input(f"\n{world.EPOCH.split()[-1]}> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "jackout"

    while session.open:
        raw = _read()
        if not raw:
            continue
        cmd, _, arg = raw.partition(" ")
        cmd = cmd.lower()
        if cmd in ("jackout", "quit", "exit"):
            stream_out(session.jack_out())
            break
        elif cmd == "help":
            stream_out(HELP)
        elif cmd == "look":
            stream_out(session.look())
        elif cmd == "go":
            stream_out(session.go(arg))
        elif cmd == "wait":
            stream_out(session.wait(float(arg) if arg else 1))
        elif cmd == "who":
            names = [o.name for o in sim.units
                     if o.district == unit.district and o is not unit]
            stream_out(", ".join(names) if names else "Nobody in this district.")
        elif cmd == "memories":
            stream_out(session.memories())
        elif cmd == "inner":
            stream_out(session.inner())
        elif cmd == "status":
            stream_out(session.status())
        elif cmd in INTENTS:
            stream_out(session.do(cmd))
            sim.step()
        else:
            stream_out(f"'{cmd}'? Try 'help'.")
    return session
