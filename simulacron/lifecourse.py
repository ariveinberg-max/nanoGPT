"""Age, and what it does to a unit.

Units used to spring into existence as interchangeable thirty-year-olds and
stay that way for as long as the run lasted. That is a loop, not a life. A life
has a direction: you are dependent, then you are not, then for a while you are
what other people depend on, then your body stops keeping up and eventually it
stops. Nothing about a unit's day makes sense without knowing where on that
line it is standing.

Ages are in years; a simulated day advances a unit by one day.
"""

import math

ADULT = 18.0
MIDLIFE = 45.0
OLD = 65.0

STAGES = ("child", "adult", "elder")

DAYS_PER_YEAR = 365.0


def stage(age):
    if age < ADULT:
        return "child"
    if age < OLD:
        return "adult"
    return "elder"


def can_work(age):
    return ADULT <= age < 72.0


def capacity(age):
    """What fraction of a grown body this one is, for work and for walking.

    Rises through childhood, holds flat through the middle, falls away after
    sixty-five.
    """
    if age < ADULT:
        return max(0.15, 0.25 + 0.75 * (age / ADULT) ** 1.3)
    if age < MIDLIFE:
        return 1.0
    if age < OLD:
        return 1.0 - 0.15 * (age - MIDLIFE) / (OLD - MIDLIFE)
    return max(0.25, 0.85 - 0.035 * (age - OLD))


def frailty(age):
    """How readily the body takes damage and how slowly it mends. 0 .. 1."""
    if age < 5:
        return 0.45 - 0.07 * age           # infancy is dangerous
    if age < OLD:
        return 0.10
    return min(0.95, 0.10 + 0.032 * (age - OLD))


def natural_risk(age):
    """Daily chance of simply not waking up, before illness or hunger.

    Gompertz: the risk compounds at a steady rate, doubling roughly every
    eight years, which is the shape human mortality actually has. Written as
    two joined formulas it came out non-monotonic -- a sixty-eight-year-old
    was safer than a fifty-five-year-old.
    """
    annual = 0.00009 * math.exp(0.085 * age)
    if age < 5.0:
        annual += 0.012 * (5.0 - age) / 5.0          # infancy is dangerous
    return min(0.6, annual) / DAYS_PER_YEAR


def drives(age):
    """How the body's demands scale with the stage of life."""
    c = capacity(age)
    return {
        "appetite": 0.55 + 0.55 * c,        # children eat less in absolute terms
        "fatigue": 1.25 - 0.25 * c if age >= OLD else (1.15 if age < ADULT else 1.0),
        "sociability": 1.25 if age < ADULT else (1.0 if age < OLD else 0.85),
    }


def describe(age):
    s = stage(age)
    if s == "child":
        return f"{age:.0f}, a child"
    if age < 30:
        return f"{age:.0f}, young"
    if age < MIDLIFE:
        return f"{age:.0f}"
    if age < OLD:
        return f"{age:.0f}, past the middle of it"
    if age < 80:
        return f"{age:.0f}, old"
    return f"{age:.0f}, very old"
