"""Cam profile definitions for cam-follower mechanisms.

This module provides classes for defining cam profiles that drive
cam-follower mechanisms. Profiles can be defined analytically using
motion laws or from discrete points with spline interpolation.

Profile Types:
    FunctionProfile: Profile from motion law + timing parameters
    PointArrayProfile: Profile from discrete points with spline interpolation

Motion Laws:
    HarmonicMotionLaw: Simple harmonic (cosine) motion
    CycloidalMotionLaw: Cycloidal motion (zero velocity/acceleration at ends)
    ModifiedTrapezoidalMotionLaw: Modified trapezoidal (low peak acceleration)
    PolynomialMotionLaw: General polynomial (3-4-5 or custom)

Factory Functions:
    polynomial_345: Create a 3-4-5 polynomial motion law
    polynomial_4567: Create a 4-5-6-7 polynomial motion law

Example:
    Create a cam profile with harmonic motion::

        from pylinkage.cam import FunctionProfile, HarmonicMotionLaw
        import math

        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi/2,
            dwell_high_end=math.pi,
            fall_end=3*math.pi/2,
        )

        # Evaluate at various angles
        r0 = profile.evaluate(0.0)       # Base radius
        r_max = profile.evaluate(math.pi) # Max radius

    Create a cam profile from discrete points::

        from pylinkage.cam import PointArrayProfile
        import math

        angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
        radii = [1.0, 1.2, 1.5, 1.5, 1.2, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
"""

from .motion_laws import (
    CycloidalMotionLaw,
    HarmonicMotionLaw,
    ModifiedTrapezoidalMotionLaw,
    MotionLaw,
    PolynomialMotionLaw,
    polynomial_345,
    polynomial_4567,
)
from .profiles import CamProfile, FunctionProfile, PointArrayProfile

__all__ = [
    # Profile base class
    "CamProfile",
    # Profile implementations
    "FunctionProfile",
    "PointArrayProfile",
    # Motion law base class
    "MotionLaw",
    # Motion law implementations
    "HarmonicMotionLaw",
    "CycloidalMotionLaw",
    "ModifiedTrapezoidalMotionLaw",
    "PolynomialMotionLaw",
    # Factory functions
    "polynomial_345",
    "polynomial_4567",
]
