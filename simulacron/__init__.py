"""Simulacron -- a working prototype.

    "So the whole thing's what, a giant computer game?"
    "No, not at all. It doesn't need a user to interact with it to function.
     Its units are fully formed self-learning cyber beings."
                                   -- The Thirteenth Floor (1999)

Fuller's first prototype recreated the era of his youth. This build is pointed
at Los Angeles in 2010 instead -- same system, later signage -- populated by
units that think, work, eat, and learn their own daily rhythm from experience.
Start it and it runs without you. Jack in and you walk around inside it while
your body stays in the lab.
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
