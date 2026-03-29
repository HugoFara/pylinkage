"""Classical mechanism synthesis methods for planar linkages.

This module implements classical analytical synthesis methods for designing
four-bar linkages to achieve specific motion requirements:

- **Function Generation**: Match input/output angle relationships using
  Freudenstein's equation.
- **Path Generation**: Synthesize linkages where a coupler point traces
  a curve through specified precision points.
- **Motion Generation**: Guide a rigid body through specified poses
  (position + orientation).
- **Burmester Theory**: The foundational geometric method underlying
  all synthesis types.

Quick Start
-----------

Path Generation (most common use case)::

    from pylinkage.synthesis import path_generation

    # Find 4-bar where coupler passes through these points
    precision_points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
    result = path_generation(precision_points)

    print(f"Found {len(result)} solutions")
    for linkage in result.solutions:
        linkage.show()

Function Generation::

    import math
    from pylinkage.synthesis import function_generation

    # Match input/output angle relationship
    angle_pairs = [
        (0, 0),
        (math.pi/6, math.pi/4),
        (math.pi/3, math.pi/2),
    ]
    result = function_generation(angle_pairs)

    if result:
        linkage = result[0]
        print(f"Crank length: {linkage.joints[2].r:.3f}")

Motion Generation::

    from pylinkage.synthesis import Pose, motion_generation

    # Guide body through these poses (x, y, angle)
    poses = [
        Pose(0, 0, 0),
        Pose(1, 1, 0.5),
        Pose(2, 0.5, 1.0),
    ]
    result = motion_generation(poses)

Working with Results
--------------------

All synthesis functions return a ``SynthesisResult`` object::

    result = path_generation(points)

    # Check if solutions were found
    if result:
        print(f"Found {len(result)} solutions")

    # Iterate over solutions
    for linkage in result:
        linkage.show()

    # Access warnings
    for warning in result.warnings:
        print(f"Warning: {warning}")

    # Access raw mathematical solutions
    for sol in result.raw_solutions:
        print(f"Crank: {sol.crank_length:.3f}")

Creating Linkages from Link Lengths
-----------------------------------

Use ``fourbar_from_lengths`` to create a four-bar directly::

    from pylinkage.synthesis import fourbar_from_lengths

    linkage = fourbar_from_lengths(
        crank_length=1.0,
        coupler_length=3.0,
        rocker_length=3.0,
        ground_length=4.0,
    )
    linkage.show()

Synthesis Theory Overview
-------------------------

**Burmester Theory**: For a set of precision positions (poses), Burmester
theory identifies all possible attachment points (circle points) on a
moving body and their corresponding fixed pivots (center points) such
that the attachment traces a circular arc during motion through the
precision positions.

**Function Generation**: Uses Freudenstein's equation to relate input
crank angle to output rocker angle. Given 3 angle pairs, there is a
unique solution. With more pairs, least-squares fitting is used.

**Path Generation**: More complex because the coupler orientation at
each point is not specified. The algorithm searches over possible
orientations using Burmester theory.

**Motion Generation**: Most constrained because both position AND
orientation are specified. Uses Burmester theory directly on the poses.

References
----------
- Freudenstein, F. "Approximate Synthesis of Four-Bar Linkages" (1955)
- McCarthy, J.M. "Geometric Design of Linkages" (2nd ed., 2011)
- Sandor & Erdman, "Advanced Mechanism Design" (1984)
- Bottema & Roth, "Theoretical Kinematics" (1979)
"""

__all__ = [
    # Type definitions
    "Point2D",
    "ComplexPoint",
    "PrecisionPoint",
    "AnglePair",
    "Pose",
    "SynthesisType",
    "FourBarSolution",
    "DyadSolution",
    # Core classes
    "SynthesisProblem",
    "SynthesisResult",
    "Dyad",
    "BurmesterCurves",
    # Main synthesis functions
    "function_generation",
    "verify_function_generation",
    "path_generation",
    "path_generation_with_timing",
    "verify_path_generation",
    "motion_generation",
    "motion_generation_3_poses",
    # Topology-aware synthesis (Phase 3)
    "NBarSolution",
    "GroupSynthesisResult",
    "QualityMetrics",
    "TopologySolution",
    "six_bar_path_generation",
    "generalized_synthesis",
    "multi_topology_synthesize",
    "nbar_solution_to_linkage",
    "compute_metrics",
    # Burmester theory functions
    "compute_pole",
    "compute_all_poles",
    "compute_circle_point_curve",
    "select_compatible_dyads",
    # Conversion functions
    "solution_to_linkage",
    "solutions_to_linkages",
    "linkage_to_synthesis_params",
    "fourbar_from_lengths",
    # Utility functions
    "grashof_check",
    "is_grashof",
    "is_crank_rocker",
    "GrashofType",
    "validate_fourbar",
    "point_to_complex",
    "complex_to_point",
]

# Type definitions
from ._types import (
    AnglePair,
    ComplexPoint,
    DyadSolution,
    FourBarSolution,
    Point2D,
    Pose,
    PrecisionPoint,
    SynthesisType,
)

# Burmester theory
from .burmester import (
    complex_to_point,
    compute_all_poles,
    compute_circle_point_curve,
    compute_pole,
    point_to_complex,
    select_compatible_dyads,
)

# Conversion functions
from .conversion import (
    fourbar_from_lengths,
    linkage_to_synthesis_params,
    nbar_solution_to_linkage,
    solution_to_linkage,
    solutions_to_linkages,
)

# Core classes
from .core import (
    BurmesterCurves,
    Dyad,
    SynthesisProblem,
    SynthesisResult,
)

# Synthesis functions
from .function_generation import function_generation, verify_function_generation
from .generalized import generalized_synthesis
from .motion_generation import motion_generation, motion_generation_3_poses
from .multi_topology import synthesize as multi_topology_synthesize
from .path_generation import path_generation, path_generation_with_timing, verify_path_generation
from .ranking import compute_metrics
from .six_bar import six_bar_path_generation

# Topology-aware synthesis (Phase 3)
from .topology_types import (
    GroupSynthesisResult,
    NBarSolution,
    QualityMetrics,
    TopologySolution,
)

# Utility functions
from .utils import (
    GrashofType,
    grashof_check,
    is_crank_rocker,
    is_grashof,
    validate_fourbar,
)
