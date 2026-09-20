"""Record a run to a trace a viewer can play back.

The simulation is Python and a browser is not, so the viewer does not drive the
simulation -- it replays one. A run is dumped here as a compact trace: a frame
an hour holding where everyone is and what they are doing, the fear each
district carries in the population's heads, where the police are looking, and
the events worth reading.

Encoded as parallel integer arrays rather than objects, because a hundred and
eighty simulated days at hourly resolution is around four thousand frames and
the difference between the two encodings is megabytes.
"""

import json

from . import lifecourse, world
from .sim import Simulation

ACTIVITIES = ("idle", "asleep", "eating", "working", "out", "resting",
              "errands", "in transit", "robbing")
EMOTIONS = ("settled", "fear", "anger", "sadness", "shame", "guilt",
            "joy", "pride", "hope", "relief")


def _q(x, scale=100):
    return max(0, min(scale, int(round(x * scale))))


def record(n_units=16, seed=5, days=180, every=4, warmup=60):
    """Run the prototype and write down what happened.

    `warmup` days run before recording starts, so the viewer opens on a
    population that has already learned its way around rather than on a
    fortnight of flailing.
    """
    sim = Simulation(n_units=n_units, seed=seed)
    sim.run_days(warmup)

    districts = list(world.DISTRICTS)
    d_index = {d: i for i, d in enumerate(districts)}
    a_index = {a: i for i, a in enumerate(ACTIVITIES)}
    e_index = {e: i for i, e in enumerate(EMOTIONS)}

    people, p_index = [], {}

    def person(u):
        if u.name not in p_index:
            p_index[u.name] = len(people)
            people.append({
                "name": u.name,
                "home": u.home.district,
                "work": u.workplace.name,
                "born": round(sim.clock.day - u.age * lifecourse.DAYS_PER_YEAR),
            })
        return p_index[u.name]

    frames, seen_log = [], 0
    total = days * world.TICKS_PER_DAY
    for tick in range(total):
        sim.step()
        if tick % every:
            continue

        ids, place, act, hunger, peril, feel, funds, age = [], [], [], [], [], [], [], []
        for u in sim.units:
            ids.append(person(u))
            place.append(d_index.get(u.district, 0))
            act.append(a_index.get(u.activity, 0))
            hunger.append(_q(u.hunger))
            peril.append(_q(u.peril()))
            feel.append(e_index.get(u.affect.dominant()[0], 0))
            funds.append(int(min(9999, u.funds)))
            age.append(int(u.age))

        # what the population believes about each district, and where the
        # police are actually looking
        fear, been = [], []
        for d in districts:
            held = [x.memory.places[d].fear for x in sim.units if d in x.memory.places]
            danger = [x.memory.places[d].danger for x in sim.units
                      if d in x.memory.places]
            fear.append(_q(sum(danger) / len(danger) if danger else 0.0))
            been.append(sum(1 for x in sim.units if x.district == d))

        events = sim.log[seen_log:]
        seen_log = len(sim.log)

        frames.append({
            "t": sim.clock.tick,
            "stamp": sim.clock.stamp(),
            "day": sim.clock.day,
            "u": ids, "d": place, "a": act, "h": hunger, "p": peril,
            "e": feel, "$": funds, "age": age,
            "fear": fear,
            "here": been,
            "patrol": [_q(sim.police.patrol.get(d, 0.0), 1000) for d in districts],
            "ev": events,
        })

    st = sim.stats()
    return {
        "meta": {
            "epoch": world.EPOCH, "seed": seed, "units": n_units,
            "days": days, "warmup": warmup, "hours_per_frame": every / 4,
            "final": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in st.items()},
        },
        "districts": districts,
        "adjacency": [[a, b, w] for (a, b), w in world._ADJACENCY.items()],
        "activities": list(ACTIVITIES),
        "emotions": list(EMOTIONS),
        "people": people,
        "dead": [{"name": u.name, "age": round(u.age), "cause": c, "tick": t}
                 for t, u, c in sim.dead],
        "frames": frames,
    }


def main(path="trace.json", **kw):
    trace = record(**kw)
    with open(path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    return trace


def build_page(out="simulacron-console.html", **kw):
    """Record a run and bake it into a standalone page.

    The viewer does not drive the simulation, so everything it needs has to be
    in the file: the template plus one run's trace, and no server.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    template = open(os.path.join(here, "viewer", "page.html")).read()
    trace = json.dumps(record(**kw), separators=(",", ":"))
    page = template.replace(
        "<script>\nconst T = window.__TRACE__;",
        '<script id="trace" type="application/json">' + trace + "</script>\n"
        "<script>\nconst T = JSON.parse("
        "document.getElementById('trace').textContent);")
    with open(out, "w") as f:
        f.write("<!doctype html><html><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,"
                "initial-scale=1,viewport-fit=cover\"></head><body>"
                + page + "</body></html>")
    return out


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "trace.json"
    t = main(out)
    print(f"{len(t['frames'])} frames, {len(t['people'])} people -> {out}")
