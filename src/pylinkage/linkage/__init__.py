"""Linkage analysis utilities (trajectory extraction, transmission,
sensitivity, tolerance).

The legacy ``Linkage`` class that previously lived here has been
removed. Use :class:`pylinkage.simulation.Linkage` for the modern
component/actuator/dyad API or :class:`pylinkage.mechanism.Mechanism`
for the links-and-joints model. The analysis helpers exported here
work with both containers via :mod:`pylinkage._compat`.
"""

__all__ = [
    "bounding_box",
    "extract_trajectories",
    "extract_trajectory",
    "kinematic_default_test",
    # Transmission angle analysis (Revolute joints)
    "TransmissionAngleAnalysis",
    "analyze_transmission",
    "compute_transmission_angle",
    # Stroke analysis (Prismatic joints)
    "StrokeAnalysis",
    "analyze_stroke",
    "compute_slide_position",
    # Sensitivity and tolerance analysis
    "SensitivityAnalysis",
    "ToleranceAnalysis",
    "analyze_sensitivity",
    "analyze_tolerance",
]

from .analysis import (
    bounding_box as bounding_box,
)
from .analysis import (
    extract_trajectories as extract_trajectories,
)
from .analysis import (
    extract_trajectory as extract_trajectory,
)
from .analysis import (
    kinematic_default_test as kinematic_default_test,
)
from .sensitivity import (
    SensitivityAnalysis as SensitivityAnalysis,
)
from .sensitivity import (
    ToleranceAnalysis as ToleranceAnalysis,
)
from .sensitivity import (
    analyze_sensitivity as analyze_sensitivity,
)
from .sensitivity import (
    analyze_tolerance as analyze_tolerance,
)
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
