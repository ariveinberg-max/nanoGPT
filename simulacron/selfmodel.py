"""What a unit believes about itself.

This is the part the previous build had none of. A unit knew its hunger and
its bank balance, which is a state, not a self: nothing in it could be wrong,
and so nothing in it could be revised. Here a unit holds beliefs about what it
is good at, where its life is going and what it is -- beliefs formed from its
own history, which reality can contradict.

The contradiction is the useful part. A unit predicts how something will go,
finds out, and the gap is surprise: it drives learning, it colours the emotion
the event produces, and a long run of things going worse than expected is what
turns competence into despair.
"""

from dataclasses import dataclass, field

DOMAINS = ("work", "social", "coping", "provision")

# Things a person could plausibly be. The old vocabulary had six entries and
# four of them were damage -- provider, outsider, victim, offender -- so the
# only positive thing on offer besides "friend" was "worker", and every unit
# in every run answered the same word. These are mostly not about work, and
# none of them is awarded: they are read off how the hours actually went.
ROLES = (
    "worker",       # time at work
    "parent",       # time spent on somebody who depends on you
    "friend",       # time spent in company
    "regular",      # the same room, often enough that they know you
    "reader",       # time at the library, the observatory
    "homebody",     # waking hours at home
    "wanderer",     # out, going nowhere in particular
    "loner",        # out, and on its own
    "patient",      # time spent unwell
    "survivor",     # came through something
    "provider",     # kept somebody fed
    "outsider",     # long enough outside it to feel outside it
    "victim",       # what was done to it
    "offender",     # what it did
)

# How a day's waking hours map onto what that makes you.
LIVED = ("worker", "parent", "friend", "regular", "reader", "homebody",
         "wanderer", "loner", "patient")

# Nobody is entirely one thing. Saturating growth only slows the approach to
# certainty -- with enough repetition a role still pins at 1.00 and the unit
# has no room to be anything else.
ROLE_CEILING = 0.92


@dataclass
class SelfModel:
    efficacy: dict = field(default_factory=lambda: {d: 0.5 for d in DOMAINS})
    roles: dict = field(default_factory=lambda: {r: 0.0 for r in ROLES})
    prospects: float = 0.0          # -1 it gets worse .. +1 it gets better
    expectation: dict = field(default_factory=dict)   # key -> predicted value
    surprises: int = 0
    close_calls: int = 0            # times it came near the end and did not

    # -- competence ---------------------------------------------------------
    def did(self, domain, success):
        """An attempt in a domain, and how it went. Belief follows evidence."""
        if domain not in self.efficacy:
            return
        self.efficacy[domain] += 0.035 * (float(success) - self.efficacy[domain])

    def can(self, domain):
        return self.efficacy.get(domain, 0.5)

    # -- identity -----------------------------------------------------------
    def endorse(self, role, amount):
        """Move a role. Growth saturates, so nothing ratchets to certainty.

        `worker` used to gain a flat amount every working tick against almost
        no decay, so it pinned at 1.00 inside three weeks and stayed there for
        the rest of the unit's life.
        """
        if role not in self.roles:
            return
        cur = self.roles[role]
        step = amount * (1.0 - cur) if amount > 0 else amount
        self.roles[role] = max(0.0, min(ROLE_CEILING, cur + step))

    def live(self, tally, rate=0.11):
        """A day's hours, and what they make of a unit.

        Identity is a readout here, not an award: whatever a unit actually did
        with its waking time is what it slowly comes to be. Everything fades,
        so a life that changes shape changes what the unit takes itself for.
        """
        for role in self.roles:
            self.roles[role] = max(0.0, self.roles[role] - 0.008)
        hours = sum(tally.get(k, 0) for k in LIVED)
        if hours <= 0:
            return
        for role in LIVED:
            share = tally.get(role, 0) / hours
            if share > 0.04:
                self.endorse(role, rate * share)

    def identity(self, age=None, status=None):
        """What this unit would call itself.

        Occupation is one fact about a person, not the whole of them, so a
        strongly held role wins over the census answer and the census answer
        is the fallback rather than the default.
        """
        if age is not None and age < 18.0:
            return "somebody's child"
        role, weight = max(self.roles.items(), key=lambda kv: kv[1])
        if weight > 0.22:
            return role
        if status:
            from .status import LABEL
            return LABEL.get(status, "nobody in particular")
        return "nobody in particular"

    # -- expectation and its violation --------------------------------------
    def expect(self, key, default=0.0):
        return self.expectation.get(key, default)

    def outcome(self, key, actual, default=0.0):
        """Report what really happened. Returns the surprise, signed."""
        predicted = self.expectation.get(key, default)
        surprise = actual - predicted
        self.expectation[key] = predicted + 0.25 * surprise
        if abs(surprise) > 0.4:
            self.surprises += 1
        self.prospects = max(-1.0, min(1.0, self.prospects + 0.02 * surprise))
        return surprise

    def survived(self, was, now):
        """Came through a bad stretch, or did not.

        This is what finally gives the `coping` belief evidence to move on.
        It was frozen at exactly 0.5 for every unit in every run, which would
        have been merely dead code except that `coping` is the courage term
        dividing fear -- so every unit was identically brave, permanently.
        """
        if was < 0.35:
            return
        if now < was - 0.15:
            self.close_calls += 1
            self.did("coping", True)
            self.endorse("victim", -0.02)
        elif now > was + 0.10:
            self.did("coping", False)

    def esteem(self):
        return (sum(self.efficacy.values()) / len(self.efficacy)
                - 0.3 * self.roles["victim"] - 0.2 * self.roles["outsider"])

    # -- readout ------------------------------------------------------------
    def narrative(self, affect=None, age=None, status=None):
        """How this unit would say its life is going, if anyone asked."""
        best = max(self.efficacy.items(), key=lambda kv: kv[1])
        worst = min(self.efficacy.items(), key=lambda kv: kv[1])
        if self.prospects > 0.15:
            arc = "thinks it is looking up"
        elif self.prospects < -0.15:
            arc = "thinks it is going badly"
        else:
            arc = "expects more of the same"
        line = (f"Sees itself as {self.identity(age, status)}; {arc}. "
                f"Good at {best[0]} ({best[1]:.2f}), "
                f"no good at {worst[0]} ({worst[1]:.2f}).")
        if affect is not None:
            if affect.trauma > 0.25:
                line += " Carrying something."
            elif affect.stress > 0.5:
                line += " Worn down."
        return line
