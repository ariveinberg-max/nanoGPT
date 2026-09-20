"""What a unit is doing with its life, in the sense a census means it.

Everyone used to have a job. `spawn` handed every working-age adult one on the
first tick and the only way out was dismissal, so the population was a hundred
per cent employed and stayed near it -- which is not what any real population
has ever looked like, and it made every unit's answer to "what are you" the
same word.

A status here is not a label attached at birth. It is a state with ways in and
ways out: people finish studying, lose work, stop looking, stay home with a
child, come to the end of what their body will do, and retire. The point is
that at any moment the city holds people doing quite different things with
their days, and that a unit's occupation is one fact about it rather than the
whole of its identity.
"""

from . import lifecourse

# Ordered roughly by how the life course runs through them.
STATUSES = ("child", "student", "employed", "unemployed", "homemaker",
            "unable", "retired")

WORKING = ("employed",)

# Roughly the shape of a city's working-age population: most in work, a
# minority out of it, some studying, some at home, some whose health has
# stopped them. Not a claim about 2010 Los Angeles specifically.
ADULT_MIX = (
    ("employed", 0.58),
    ("unemployed", 0.11),
    ("student", 0.09),
    ("homemaker", 0.13),
    ("unable", 0.05),
    ("retired", 0.04),
)

LABEL = {
    "child": "a child",
    "student": "studying",
    "employed": "in work",
    "unemployed": "out of work",
    "homemaker": "keeping a house",
    "unable": "not able to work",
    "retired": "retired",
}


def initial(age, rng):
    """What somebody this age is plausibly doing when the prototype opens."""
    if age < lifecourse.ADULT:
        return "child"
    if age >= 68:
        return "retired"
    if age < 24 and rng.random() < 0.45:
        return "student"
    if age >= 62 and rng.random() < 0.35:
        return "retired"
    keys = [k for k, _ in ADULT_MIX]
    weights = [w for _, w in ADULT_MIX]
    total = sum(weights)
    draw = rng.random() * total
    for k, w in zip(keys, weights):
        draw -= w
        if draw <= 0:
            return k
    return "employed"


def can_hold_job(status):
    return status in WORKING


def seeks_work(status):
    return status == "unemployed"


def transitions(u, rng, has_young_dependent, partner_working):
    """Where this unit's status could move today, and why.

    Returns (new_status, reason) or None. Kept as one place so the life
    course is legible rather than scattered through the daily pass.

    Every move is damped by how long the unit has been where it is. Without
    that, a status was a coin flipped daily: homemakers, retirees and the
    long-term unwell all drained back into work inside a season and the city
    was a hundred per cent employed again. People do not leave a settled life
    because one day went badly.
    """
    s = u.status
    age = u.age
    # inertia: a life gets harder to change the longer you have been living it
    settled = 1.0 / (1.0 + u.status_days / 70.0)

    def roll(p):
        return rng.random() < p * settled

    if s == "child":
        if age >= lifecourse.ADULT:
            return ("student", "old enough to be let out into it") \
                if rng.random() < 0.4 else ("unemployed", "old enough to work")
        return None

    if age >= 70 and s != "retired":
        return "retired", "past what the work will take"
    if s != "retired" and not lifecourse.can_work(age):
        return "retired", "past what the work will take"

    # a body that has stopped keeping up
    if s in ("employed", "unemployed", "student") and u.health < 0.45 \
            and u.pain > 0.45 and roll(0.05):
        return "unable", "not well enough for it any more"
    # coming back from that takes a long stretch of being well, not one day
    if s == "unable" and u.health > 0.85 and u.pain < 0.1 and roll(0.006):
        return "unemployed", "well enough to look again"

    # walking out. People leave work for reasons that are not dismissal, and
    # a city where the only exit from employment is being fired ends up a
    # hundred per cent employed however it started.
    if s == "employed" and u.person is not None:
        strain = u.affect.stress
        loose = 1.0 - u.person.scale("conscientiousness")
        pull = u.person.holds("freedom") + 0.6 * u.person.holds("pleasure")
        if strain > 0.55 and roll(0.05 * (0.4 + loose) * (0.5 + pull)):
            return "unemployed", "could not keep doing it"

    if s == "student" and age >= 22 and roll(0.02):
        return "unemployed", "done with studying"

    # somebody has to be at home
    if s in ("employed", "unemployed") and has_young_dependent \
            and partner_working and roll(0.03):
        return "homemaker", "somebody had to be at home"
    if s == "homemaker" and not has_young_dependent \
            and u.funds < 4 * 18.0 and roll(0.012):
        return "unemployed", "the money ran out"

    # giving up on looking, and picking it back up
    if s == "unemployed" and u.affect.stress > 0.7 \
            and u.selfmodel.can("work") < 0.3 and roll(0.015):
        return "unable", "stopped looking"

    if s == "retired" and age < 70 and u.funds < 18.0 and roll(0.008):
        return "unemployed", "cannot afford to be retired"

    return None
