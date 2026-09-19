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

ROLES = ("worker", "provider", "friend", "outsider", "victim", "offender")


@dataclass
class SelfModel:
    efficacy: dict = field(default_factory=lambda: {d: 0.5 for d in DOMAINS})
    roles: dict = field(default_factory=lambda: {r: 0.0 for r in ROLES})
    prospects: float = 0.0          # -1 it gets worse .. +1 it gets better
    expectation: dict = field(default_factory=dict)   # key -> predicted value
    surprises: int = 0

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
        if role in self.roles:
            self.roles[role] = max(0.0, min(1.0, self.roles[role] + amount))

    def identity(self):
        role, weight = max(self.roles.items(), key=lambda kv: kv[1])
        return role if weight > 0.25 else "nobody in particular"

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

    def esteem(self):
        return (sum(self.efficacy.values()) / len(self.efficacy)
                - 0.3 * self.roles["victim"] - 0.2 * self.roles["outsider"])

    # -- readout ------------------------------------------------------------
    def narrative(self, affect=None):
        """How this unit would say its life is going, if anyone asked."""
        best = max(self.efficacy.items(), key=lambda kv: kv[1])
        worst = min(self.efficacy.items(), key=lambda kv: kv[1])
        if self.prospects > 0.15:
            arc = "thinks it is looking up"
        elif self.prospects < -0.15:
            arc = "thinks it is going badly"
        else:
            arc = "expects more of the same"
        line = (f"Sees itself as {self.identity()}; {arc}. "
                f"Good at {best[0]} ({best[1]:.2f}), "
                f"no good at {worst[0]} ({worst[1]:.2f}).")
        if affect is not None:
            if affect.trauma > 0.25:
                line += " Carrying something."
            elif affect.stress > 0.5:
                line += " Worn down."
        return line
