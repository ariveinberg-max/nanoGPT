"""What a unit remembers, and what remembering does to it.

The previous build kept a list of memories that nothing ever read: units
recorded their lives and could not consult them. Here recall is part of the
decision loop. Before a unit chooses what to do it cues its memory with where
it is, who is present and what it is thinking of doing, and what comes back
colours the choice -- a street where something happened to you is not the same
street afterwards.

Two stores, doing different jobs:

* Episodes: particular things that happened, encoded at a strength set by how
  much they mattered emotionally. Ordinary ones fade. The worst ones do not,
  and can surface unbidden.
* Beliefs: what many episodes have added up to. A unit that has been hurt in
  Boyle Heights twice does not reason from the two occasions; it believes the
  place is dangerous, and that belief is what reaches the decision.
"""

from dataclasses import dataclass, field

HALF_LIFE = 2400.0      # ticks until an ordinary memory is half as reachable
CAPACITY = 400
VIVID = 0.22            # below this an episode cannot frighten anyone


@dataclass
class Episode:
    tick: int
    kind: str                       # "robbed", "worked", "turned away", ...
    text: str
    place: str = ""                 # venue key
    district: str = ""
    people: tuple = ()
    emotions: dict = field(default_factory=dict)
    salience: float = 0.1           # encoding strength
    recalls: int = 0
    because: str = ""               # what this happened on the back of
    drift: float = 0.0              # how far it has moved from what happened

    def intensity(self):
        return sum(self.emotions.values())

    def retrievability(self, now):
        """How reachable this is. Strong memories stop fading; that is trauma."""
        age = max(0, now - self.tick)
        recency = 0.5 ** (age / HALF_LIFE)
        floor = self.salience ** 2
        return self.salience * max(recency, floor)


def _reconstruct(ep, mood, stress):
    """Rebuild the memory instead of replaying it.

    A recalled episode is assembled from fragments, and what fills the gaps is
    how the unit feels now. So a memory shifts a little each time it is called
    up, toward the present mood, and faster under stress -- which is why a bad
    stretch makes the whole of the past look worse, and why a unit's account of
    an old event stops matching what the simulation recorded happening.
    """
    if mood is None:
        return
    # Vivid episodes resist being rebuilt; forgettable ones are mostly gaps
    # already. And drift saturates -- past a point a memory is a feeling with
    # a caption, and there is nothing left to distort.
    rate = (0.04 + 0.10 * stress) * (1.0 - 0.6 * ep.salience) * (1.0 - ep.drift)
    for e, now in mood.items():
        was = ep.emotions.get(e, 0.0)
        if was or now > 0.05:
            ep.emotions[e] = max(0.0, min(1.0, was + rate * (now - was)))
    ep.drift = min(1.0, ep.drift + rate)


@dataclass
class PlaceBelief:
    danger: float = 0.0             # 0 safe .. 1 something bad happens here
    warmth: float = 0.0             # 0 nothing good .. 1 good things happen here
    fear: float = 0.0               # the dread the place calls up, consolidated
    visits: int = 0

    def cool(self, rate=1.0):
        """Nothing happening here is itself evidence about the place.

        Danger never decayed, so a belief once raised stayed raised for the
        rest of a unit's life and fear could only ever accumulate. Somewhere
        you have walked through a hundred times without incident stops feeling
        dangerous, slowly.
        """
        # Tuned against the crime rate: too fast and beliefs drain between
        # incidents so every district reads equally safe, too slow and they
        # only ever accumulate and every district reads equally dangerous.
        self.danger = max(0.0, self.danger - 0.0008 * rate)
        self.fear = max(0.0, self.fear - 0.0011 * rate)

    def update(self, valence, harm, fear=0.0):
        self.visits += 1
        rate = 1.0 / min(self.visits, 12)
        self.danger += rate * (harm - self.danger)
        self.warmth += rate * (max(0.0, valence) - self.warmth)
        # Dread accrues fast and leaves slowly: one bad night on a street is
        # enough, and a hundred uneventful ones only partly undo it.
        self.fear = max(fear, self.fear - 0.004) if fear < self.fear \
            else self.fear + 0.55 * (fear - self.fear)


class Memory:
    def __init__(self):
        self.episodes = []
        self.by_district = {}       # district -> [episode], so recall is local
        self.vivid = {}             # district -> [episode] worth being afraid of
        self.places = {}            # district -> PlaceBelief
        self.intrusion = None       # an episode that surfaced on its own
        self._dread = {}            # (tick, district, place) -> value
        self._dread_tick = -1

    # -- encoding -----------------------------------------------------------
    def encode(self, ep, novelty=0.0):
        """Store an episode. What you felt is what you keep."""
        ep.salience = min(1.0, 0.08 + 0.75 * ep.intensity() + 0.2 * novelty)
        self.episodes.append(ep)
        self.by_district.setdefault(ep.district, []).append(ep)
        if ep.salience >= VIVID:
            self.vivid.setdefault(ep.district, []).append(ep)
        self._dread.clear()
        if ep.district:
            b = self.places.setdefault(ep.district, PlaceBelief())
            harm = ep.emotions.get("fear", 0.0) + ep.emotions.get("anger", 0.0)
            b.update(ep.intensity() if not harm else -harm, min(1.0, harm),
                     fear=ep.emotions.get("fear", 0.0))
        if len(self.episodes) > CAPACITY:
            # forget the least reachable, not the oldest
            self.episodes.sort(key=lambda e: e.retrievability(ep.tick))
            del self.episodes[:len(self.episodes) - CAPACITY]
            self.episodes.sort(key=lambda e: e.tick)
            self.by_district, self.vivid = {}, {}
            for e in self.episodes:
                self.by_district.setdefault(e.district, []).append(e)
                if e.salience >= VIVID:
                    self.vivid.setdefault(e.district, []).append(e)

    # -- retrieval ----------------------------------------------------------
    @staticmethod
    def _match(ep, district="", place="", people=(), kind=""):
        score = 0.0
        if district and ep.district == district:
            score += 0.35
        if place and ep.place == place:
            score += 0.30
        if kind and ep.kind == kind:
            score += 0.20
        if people and set(people) & set(ep.people):
            score += 0.40
        return min(1.0, score)

    def recall(self, now, k=4, rehearse=False, vivid_only=False, mood=None,
               stress=0.0, **cue):
        """The memories this situation brings up, strongest first.

        `rehearse` only for deliberate recall. Evaluating options cues memory
        several times an hour, and letting that strengthen what it touched
        drove every trivial episode to maximum salience within days -- a unit's
        memory filled up with equally unforgettable lunches.
        """
        # Scanning every episode for every option was well over half the
        # simulation's running time. A cue that names a district only needs
        # the episodes from that district.
        index = self.vivid if vivid_only else self.by_district
        pool = index.get(cue["district"], ()) if cue.get("district") \
            else self.episodes
        scored = []
        for ep in pool:
            m = self._match(ep, **cue)
            if m <= 0:
                continue
            scored.append((m * ep.retrievability(now), ep))
        scored.sort(key=lambda t: -t[0])
        out = []
        for strength, ep in scored[:k]:
            ep.recalls += 1
            if rehearse:
                ep.salience = min(1.0, ep.salience + 0.002)
                _reconstruct(ep, mood, stress)
            out.append((strength, ep))
        return out

    def dread(self, now, **cue):
        """How much fear this place calls up. Consolidated, not recomputed.

        This used to scan episodes on every option of every hour and was most
        of the simulation's running time. It is also the wrong model: nobody
        re-derives their feeling about a street from the particular nights they
        spent on it. You retrieve the feeling. The episodes are still there,
        and still reachable by `recall`, for the things that genuinely need
        them -- intrusive memory, being asked, meeting someone again.
        """
        b = self.places.get(cue.get("district", ""))
        return b.fear if b else 0.0

    def intrude(self, now, rng, trauma):
        """The worst things come back on their own. Returns an episode or None."""
        self.intrusion = None
        if trauma < 0.1:
            return None
        worst = [e for e in self.episodes if e.salience > 0.6
                 and e.emotions.get("fear", 0) + e.emotions.get("shame", 0)
                 + e.emotions.get("guilt", 0) + e.emotions.get("sadness", 0) > 0.5]
        if not worst or rng.random() > 0.02 + 0.10 * trauma:
            return None
        ep = worst[int(rng.integers(len(worst)))]
        ep.recalls += 1
        self.intrusion = ep
        return ep

    # -- readouts -----------------------------------------------------------
    def danger_of(self, district):
        b = self.places.get(district)
        return b.danger if b else 0.0

    def recent(self, n=6):
        return self.episodes[-n:]

    def describe_places(self):
        rows = sorted(self.places.items(), key=lambda kv: -kv[1].danger)
        return [f"{d}: danger {b.danger:.2f} warmth {b.warmth:.2f} ({b.visits} visits)"
                for d, b in rows]
