"""Definition of joints."""

__all__ = ["Crank", "Fixed", "Linear", "Prismatic", "Revolute", "Static"]

from .crank import Crank as Crank
from .fixed import Fixed as Fixed
from .joint import Static as Static
from .prismatic import Linear as Linear  # Deprecated alias
from .prismatic import Prismatic as Prismatic
from .revolute import Revolute as Revolute
