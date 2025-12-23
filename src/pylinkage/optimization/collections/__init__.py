"""
Package for collection objects.
"""

__all__ = ["Agent", "MutableAgent", "ParetoFront", "ParetoSolution"]

from .agent import Agent as Agent
from .mutable_agent import MutableAgent as MutableAgent
from .pareto import ParetoFront as ParetoFront
from .pareto import ParetoSolution as ParetoSolution
