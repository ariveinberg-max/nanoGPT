"""A local server that runs the prototype and lets you watch it live.

The recording in `record.py` was a compromise: the simulation is Python and a
browser is not, so the console replayed a trace. This does the other thing.
The prototype runs here, in a thread, and the page asks it what is happening.

Which buys the part a trace could never carry. A recording can hold where
everybody is; it cannot hold what any of them believes, remembers, or would
say about itself, because that is megabytes per unit per hour. Live, you can
ask -- click anyone and the server hands back that unit's memories ranked by
what they cost it, what it believes about each district, who it knows and what
it thinks of them, what it is attending to right now, what it says about why
it did the last thing it did, and what actually decided it.

Standard library only. No dependencies, no build step.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os.path import abspath, dirname, join
from urllib.parse import parse_qs, urlparse

from . import goals as goals_mod, lifecourse, status as status_mod, world
from .sim import Simulation
from .workspace import confabulate, honest_reason, self_report

HERE = dirname(abspath(__file__))


class Engine:
    """The prototype, running on its own clock, guarded by one lock."""

    def __init__(self, n_units=20, seed=5, warmup=90, hz=8.0):
        self.lock = threading.Lock()
        self.sim = Simulation(n_units=n_units, seed=seed)
        self.hz = hz
        self.playing = True
        self.ready = False
        self.warmup = warmup
        self.seen = 0
        self.events = []
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        with self.lock:
            self.sim.run_days(self.warmup)
            self.ready = True
        self._run()

    def _run(self):
        while True:
            if not self.playing:
                time.sleep(0.05)
                continue
            started = time.perf_counter()
            with self.lock:
                self.sim.step()
                new = self.sim.log[self.seen:]
                self.seen = len(self.sim.log)
                if new:
                    stamp = self.sim.clock.stamp()
                    self.events = (self.events + [{"when": stamp, "text": e}
                                                  for e in new])[-120:]
            time.sleep(max(0.0, 1.0 / self.hz - (time.perf_counter() - started)))

    # ---------------------------------------------------------------- reads
    def state(self):
        with self.lock:
            s = self.sim
            if not self.ready:
                return {"booting": True, "warmup": self.warmup,
                        "day": s.clock.day}
            districts = []
            for d in world.DISTRICTS:
                danger = [u.memory.places[d].danger for u in s.units
                          if d in u.memory.places]
                districts.append({
                    "name": d,
                    "fear": round(sum(danger) / len(danger), 3) if danger else 0.0,
                    "here": sum(1 for u in s.units if u.district == d),
                    "patrol": round(s.police.patrol.get(d, 0.0), 4),
                    "offences": s.police.offences.get(d, 0),
                    "arrests": s.police.arrests.get(d, 0),
                })
            units = [{
                "name": u.name, "district": u.district,
                "activity": u.activity, "age": int(u.age),
                "hunger": round(u.hunger, 2), "peril": round(u.peril(), 2),
                "funds": int(u.funds),
                "feeling": u.affect.dominant()[0],
                "stress": round(u.affect.stress, 2),
                "child": u.age < lifecourse.ADULT,
                "status": u.status,
                "thought": u.thought,
            } for u in s.units]
            return {
                "booting": False,
                "epoch": world.EPOCH, "stamp": s.clock.stamp(), "day": s.clock.day,
                "playing": self.playing, "hz": self.hz,
                "districts": districts, "units": units,
                "events": list(reversed(self.events[-60:])),
                "stats": {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in s.stats().items()},
                "dead": [{"name": u.name, "age": int(u.age), "cause": c}
                         for _, u, c in s.dead[-8:]],
            }

    def unit(self, name):
        """Everything one unit is carrying. This is what a trace cannot hold."""
        with self.lock:
            u = self.sim.by_name.get(name)
            if u is None:
                return {"gone": True, "name": name}
            return {
                "gone": False,
                "name": u.name, "age": round(u.age, 1),
                "stage": lifecourse.describe(u.age),
                "where": (u.location.name if u.location else "in transit"),
                "district": u.district, "activity": u.activity,
                "home": u.home.name, "work": u.workplace.name,
                "employed": u.employed,
                "body": {"hunger": round(u.hunger, 2), "fatigue": round(u.fatigue, 2),
                         "loneliness": round(u.loneliness, 2),
                         "pain": round(u.pain, 2), "health": round(u.health, 2),
                         "peril": round(u.peril(), 2), "funds": round(u.funds, 2),
                         "debt": round(u.debt, 2)},
                "affect": {k: round(v, 2) for k, v in u.affect.intensity.items()
                           if v > 0.02},
                "stress": round(u.affect.stress, 2),
                "trauma": round(u.affect.trauma, 2),
                "drive": round(u.affect.drive(), 2),
                "effort": round(u.effort.available(), 2),
                "self": u.selfmodel.narrative(u.affect, u.age, u.status),
                "status": u.status,
                "doing": status_mod.LABEL.get(u.status, u.status),
                "personality": (u.person.five if u.person else {}),
                "is": (u.person.sketch() if u.person else ""),
                "values": list(u.person.values) if u.person else [],
                "interests": list(u.person.interests) if u.person else [],
                "quirk": (u.person.quirk if u.person else ""),
                "after": goals_mod.describe(u),
                "efficacy": {k: round(v, 2) for k, v in u.selfmodel.efficacy.items()},
                "roles": {k: round(v, 2) for k, v in u.selfmodel.roles.items()
                          if v > 0.02},
                "worldview": u.worldview.describe(),
                "confidence": round(u.worldview.confidence, 2),
                "family": u.family.describe(),
                "attending": (u.workspace.current.said()
                              if u.workspace.current else "nothing much"),
                "says": self_report(u),
                "its_reason": confabulate(u),
                "the_reason": honest_reason(u),
                "thought": u.thought, "why_thought": u.why_thought,
                "habits": u.habits.size(),
                "knows_places": len(u.known_places),
                "people": [{
                    "name": r.name,
                    "familiarity": round(r.familiarity, 2),
                    "trust": round(r.trust, 2),
                    "grudge": round(r.grudge, 2),
                    "thinks": r.mind.describe(),
                } for r in sorted(u.social.people.values(),
                                  key=lambda r: -r.familiarity)[:8]],
                "places": [{"name": d,
                            "danger": round(b.danger, 2),
                            "warmth": round(b.warmth, 2),
                            "visits": b.visits}
                           for d, b in sorted(u.memory.places.items(),
                                              key=lambda kv: -kv[1].danger)],
                "memories": [{
                    "text": e.text, "salience": round(e.salience, 2),
                    "drift": round(e.drift, 2), "because": e.because,
                    "felt": {k: round(v, 2) for k, v in
                             sorted(e.emotions.items(), key=lambda kv: -kv[1])[:2]},
                } for e in sorted(u.memory.episodes, key=lambda e: -e.salience)[:10]],
                "considered": [{
                    "option": o.label(), "score": round(o.score, 2),
                    "why": o.why(4),
                } for o in sorted(u.considered or [], key=lambda o: -o.score)[:6]],
            }

    def control(self, q):
        if "play" in q:
            self.playing = q["play"][0] not in ("0", "false")
        if "hz" in q:
            try:
                self.hz = max(0.5, min(240.0, float(q["hz"][0])))
            except ValueError:
                pass
        if "step" in q:
            with self.lock:
                for _ in range(int(q["step"][0] or 1)):
                    self.sim.step()
        return {"playing": self.playing, "hz": self.hz}


def make_handler(engine, page):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body, ctype="application/json"):
            raw = body if isinstance(body, bytes) else body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._send(json.dumps(engine.state()))
            if url.path == "/api/unit":
                return self._send(json.dumps(engine.unit(q.get("name", [""])[0])))
            if url.path == "/api/control":
                return self._send(json.dumps(engine.control(q)))
            self.send_error(404)

        def log_message(self, *a):
            pass                        # the console is the output, not the log

    return Handler


def serve(host="127.0.0.1", port=8000, n_units=20, seed=5, warmup=90, hz=8.0):
    page = open(join(HERE, "viewer", "live.html")).read()
    page = ("<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "</head><body>" + page + "</body></html>")
    engine = Engine(n_units=n_units, seed=seed, warmup=warmup, hz=hz)
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, page))
    return httpd, engine
