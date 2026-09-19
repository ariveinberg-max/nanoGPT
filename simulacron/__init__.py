"""Simulacron -- a working prototype.

    "So the whole thing's what, a giant computer game?"
    "No, not at all. It doesn't need a user to interact with it to function.
     Its units are fully formed self-learning cyber beings."
                                   -- The Thirteenth Floor (1999)

Los Angeles, circa 1937, populated by units that think, work, eat, and learn
their own daily rhythm from experience. Start it and it runs without you. Jack
in and you walk around inside it while your body stays in the lab.
"""

from .brain import INTENTS, Learner, Policy
from .link import LinkSession, repl
from .sim import Simulation
from .unit import Unit
from .world import Clock, DISTRICTS, VENUES, EPOCH

__all__ = [
    "Simulation", "Unit", "Policy", "Learner", "LinkSession", "repl",
    "Clock", "DISTRICTS", "VENUES", "EPOCH", "INTENTS",
]
