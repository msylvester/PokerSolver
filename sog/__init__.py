"""
Student of Games: A unified learning algorithm for both perfect and imperfect information games.

This package implements the core algorithm from the paper:
"Student of Games: A unified learning algorithm for both perfect and imperfect information games"
"""

from .core import StudentOfGames, PublicBeliefState
from .cfr import GrowingTreeCFR
from .network import CounterfactualValuePolicyNetwork
from .selfplay import SoundSelfPlay

__version__ = "0.1.0"
__all__ = [
    "StudentOfGames",
    "PublicBeliefState", 
    "GrowingTreeCFR",
    "CounterfactualValuePolicyNetwork",
    "SoundSelfPlay",
]