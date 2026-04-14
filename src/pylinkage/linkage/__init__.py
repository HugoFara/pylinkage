"""
Definition and analysis of a linkage as a dynamic set of joints.
"""

__all__ = [
    "Linkage",
    "Simulation",
    "bounding_box",
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
    extract_trajectory as extract_trajectory,
)
from .analysis import (
    kinematic_default_test as kinematic_default_test,
)
from .linkage import Linkage as Linkage
from .linkage import Simulation as Simulation
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
