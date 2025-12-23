"""
Definition and analysis of a linkage as a dynamic set of joints.
"""

__all__ = [
    "Linkage",
    "Simulation",
    "bounding_box",
    "kinematic_default_test",
    # Transmission angle analysis (Revolute joints)
    "TransmissionAngleAnalysis",
    "analyze_transmission",
    "compute_transmission_angle",
    # Stroke analysis (Prismatic joints)
    "StrokeAnalysis",
    "analyze_stroke",
    "compute_slide_position",
]

from .analysis import (
    bounding_box as bounding_box,
)
from .analysis import (
    kinematic_default_test as kinematic_default_test,
)
from .linkage import Linkage as Linkage
from .linkage import Simulation as Simulation
from .transmission import (
    StrokeAnalysis as StrokeAnalysis,
)
from .transmission import (
    TransmissionAngleAnalysis as TransmissionAngleAnalysis,
)
from .transmission import (
    analyze_stroke as analyze_stroke,
)
from .transmission import (
    analyze_transmission as analyze_transmission,
)
from .transmission import (
    compute_slide_position as compute_slide_position,
)
from .transmission import (
    compute_transmission_angle as compute_transmission_angle,
)
