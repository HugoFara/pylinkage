"""Cam profile definitions.

A cam profile defines the displacement as a function of cam angle.
Profiles support evaluation, derivatives, and integration with the
optimization constraint system.

Available profiles:
    FunctionProfile: Profile from motion law + timing parameters
    PointArrayProfile: Profile from discrete points with spline interpolation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ._numba_core import (
    PROFILE_SPLINE,
    compute_pitch_radius,
    evaluate_profile_derivative,
    evaluate_profile_displacement,
    evaluate_spline_profile_derivative,
    evaluate_spline_profile_displacement,
)
from .motion_laws import HarmonicMotionLaw, MotionLaw

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CamProfile(ABC):
    """Abstract base class for cam profiles.

    A cam profile defines the follower displacement as a function of
    cam rotation angle. The profile determines the shape of the cam
    and the resulting follower motion.

    Attributes:
        base_radius: Minimum radius (base circle radius).
        name: Human-readable identifier.
    """

    base_radius: float
    name: str

    @abstractmethod
    def evaluate(self, angle: float) -> float:
        """Evaluate cam radius at given angle.

        Args:
            angle: Cam rotation angle in radians.

        Returns:
            Cam radius at this angle.
        """
        ...

    @abstractmethod
    def evaluate_derivative(self, angle: float) -> float:
        """Evaluate cam radius derivative (dr/dtheta) at given angle.

        Required for pressure angle and pitch curve calculations.

        Args:
            angle: Cam rotation angle in radians.

        Returns:
            Derivative dr/dtheta at this angle.
        """
        ...

    def pitch_radius(self, angle: float, roller_radius: float) -> float:
        """Compute pitch curve radius for roller follower.

        The pitch curve is the path traced by the roller center.

        Args:
            angle: Cam rotation angle in radians.
            roller_radius: Radius of the roller follower.

        Returns:
            Distance from cam center to roller center.
        """
        cam_radius = self.evaluate(angle)
        cam_derivative = self.evaluate_derivative(angle)
        return float(compute_pitch_radius(cam_radius, cam_derivative, roller_radius))

    def pressure_angle(self, angle: float) -> float:
        """Compute pressure angle at given cam angle.

        The pressure angle is the angle between the normal to the
        cam profile and the direction of follower motion. High
        pressure angles (> 30 degrees) can cause binding.

        Args:
            angle: Cam rotation angle in radians.

        Returns:
            Pressure angle in radians.
        """
        cam_radius = self.evaluate(angle)
        cam_derivative = self.evaluate_derivative(angle)
        if abs(cam_radius) < 1e-10:
            return 0.0
        return math.atan(cam_derivative / cam_radius)

    @abstractmethod
    def get_constraints(self) -> tuple[float, ...]:
        """Return optimizable constraint values.

        Returns:
            Tuple of constraint values that can be optimized.
        """
        ...

    @abstractmethod
    def set_constraints(self, *args: float | None) -> None:
        """Set constraint values from optimization.

        Args:
            *args: Constraint values to set.
        """
        ...

    @abstractmethod
    def to_numba_data(self) -> tuple[int, NDArray[np.float64]]:
        """Convert profile to numba-compatible representation.

        Returns:
            Tuple of (profile_type_code, data_array).
        """
        ...


class FunctionProfile(CamProfile):
    """Cam profile from motion law and timing parameters.

    Defines a rise-dwell-fall-dwell motion using a standard motion law.
    The motion law determines the shape of the rise and fall segments.

    Attributes:
        motion_law: The motion law defining displacement curve.
        base_radius: Base circle radius.
        total_lift: Maximum follower displacement.
        rise_start: Angle where rise begins (radians).
        rise_end: Angle where rise ends / dwell-high begins.
        dwell_high_end: Angle where dwell-high ends / fall begins.
        fall_end: Angle where fall ends / dwell-low begins.

    Example:
        >>> from pylinkage.cam import FunctionProfile, HarmonicMotionLaw
        >>> import math
        >>> profile = FunctionProfile(
        ...     motion_law=HarmonicMotionLaw(),
        ...     base_radius=1.0,
        ...     total_lift=0.5,
        ...     rise_start=0.0,
        ...     rise_end=math.pi/2,
        ...     dwell_high_end=math.pi,
        ...     fall_end=3*math.pi/2,
        ... )
        >>> profile.evaluate(0.0)  # At base radius
        1.0
        >>> profile.evaluate(math.pi)  # At max lift
        1.5
    """

    __slots__ = (
        "motion_law",
        "base_radius",
        "total_lift",
        "rise_start",
        "rise_end",
        "dwell_high_end",
        "fall_end",
        "name",
    )

    motion_law: MotionLaw
    total_lift: float
    rise_start: float
    rise_end: float
    dwell_high_end: float
    fall_end: float

    def __init__(
        self,
        motion_law: MotionLaw | None = None,
        base_radius: float = 1.0,
        total_lift: float = 0.5,
        rise_start: float = 0.0,
        rise_end: float | None = None,
        dwell_high_end: float | None = None,
        fall_end: float | None = None,
        name: str | None = None,
    ) -> None:
        """Create a function-based cam profile.

        Args:
            motion_law: Motion law for rise/fall segments.
                       Defaults to HarmonicMotionLaw.
            base_radius: Base circle radius.
            total_lift: Maximum follower displacement.
            rise_start: Angle where rise begins (radians).
            rise_end: Angle where rise ends. Defaults to pi/2.
            dwell_high_end: Angle where dwell-high ends. Defaults to pi.
            fall_end: Angle where fall ends. Defaults to 3*pi/2.
            name: Human-readable identifier.
        """
        if motion_law is None:
            motion_law = HarmonicMotionLaw()

        self.motion_law = motion_law
        self.base_radius = base_radius
        self.total_lift = total_lift
        self.rise_start = rise_start
        self.rise_end = rise_end if rise_end is not None else math.pi / 2
        self.dwell_high_end = dwell_high_end if dwell_high_end is not None else math.pi
        self.fall_end = fall_end if fall_end is not None else 3 * math.pi / 2
        self.name = name if name is not None else f"FunctionProfile-{id(self)}"

    def evaluate(self, angle: float) -> float:
        """Evaluate cam radius at given angle."""
        coeffs = self.motion_law.to_numba_coefficients()
        return float(
            evaluate_profile_displacement(
                angle,
                self.motion_law.profile_type,
                self.base_radius,
                self.total_lift,
                self.rise_start,
                self.rise_end,
                self.dwell_high_end,
                self.fall_end,
                coeffs,
            )
        )

    def evaluate_derivative(self, angle: float) -> float:
        """Evaluate cam radius derivative at given angle."""
        coeffs = self.motion_law.to_numba_coefficients()
        return float(
            evaluate_profile_derivative(
                angle,
                self.motion_law.profile_type,
                self.base_radius,
                self.total_lift,
                self.rise_start,
                self.rise_end,
                self.dwell_high_end,
                self.fall_end,
                coeffs,
            )
        )

    def get_constraints(self) -> tuple[float, ...]:
        """Return optimizable constraints.

        Returns base_radius and total_lift as optimizable parameters.
        Timing parameters are typically fixed.
        """
        return (self.base_radius, self.total_lift)

    def set_constraints(
        self,
        base_radius: float | None = None,
        total_lift: float | None = None,
        *args: float | None,
    ) -> None:
        """Set constraint values."""
        if base_radius is not None:
            self.base_radius = base_radius
        if total_lift is not None:
            self.total_lift = total_lift

    def to_numba_data(self) -> tuple[int, NDArray[np.float64]]:
        """Convert to numba-compatible representation."""
        coeffs = self.motion_law.to_numba_coefficients()
        # Pack timing parameters and coefficients into single array
        data = np.concatenate(
            [
                np.array(
                    [
                        self.base_radius,
                        self.total_lift,
                        self.rise_start,
                        self.rise_end,
                        self.dwell_high_end,
                        self.fall_end,
                    ],
                    dtype=np.float64,
                ),
                coeffs,
            ]
        )
        return (self.motion_law.profile_type, data)


class PointArrayProfile(CamProfile):
    """Cam profile from discrete (angle, radius) points.

    Uses cubic spline interpolation to create a smooth profile
    through the given points. Supports periodic boundary conditions
    for closed cam profiles.

    The spline coefficients are precomputed at construction for
    efficient numba evaluation in the simulation loop.

    Example:
        >>> import math
        >>> angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
        >>> radii = [1.0, 1.2, 1.5, 1.5, 1.2, 1.0]
        >>> profile = PointArrayProfile(angles=angles, radii=radii)
        >>> profile.evaluate(math.pi/4)  # Approximately 1.2
        1.2
    """

    __slots__ = (
        "base_radius",
        "_angles",
        "_radii",
        "_spline_coeffs",
        "name",
    )

    _angles: NDArray[np.float64]
    _radii: NDArray[np.float64]
    _spline_coeffs: NDArray[np.float64]

    def __init__(
        self,
        angles: list[float] | NDArray[np.float64],
        radii: list[float] | NDArray[np.float64],
        periodic: bool = True,
        name: str | None = None,
    ) -> None:
        """Create a point array cam profile.

        Args:
            angles: Angle values in radians (must be sorted ascending).
            radii: Corresponding radius values.
            periodic: If True, enforce periodic boundary (r(0) = r(2*pi)).
            name: Human-readable identifier.

        Raises:
            ValueError: If angles and radii have different lengths.
            ValueError: If fewer than 3 points provided.
        """
        angles_arr = np.array(angles, dtype=np.float64)
        radii_arr = np.array(radii, dtype=np.float64)

        if len(angles_arr) != len(radii_arr):
            raise ValueError("angles and radii must have same length")
        if len(angles_arr) < 3:
            raise ValueError("Need at least 3 points for spline interpolation")

        self._angles = angles_arr
        self._radii = radii_arr
        self.base_radius = float(np.min(radii_arr))
        self.name = name if name is not None else f"PointArrayProfile-{id(self)}"

        # Compute spline coefficients
        self._spline_coeffs = self._compute_spline_coefficients(periodic)

    def _compute_spline_coefficients(self, periodic: bool) -> NDArray[np.float64]:
        """Compute cubic spline coefficients.

        Uses scipy.interpolate.CubicSpline for robust computation,
        then extracts coefficients for numba evaluation.

        Args:
            periodic: Whether to use periodic boundary conditions.

        Returns:
            Spline coefficients array of shape (n_segments, 5)
            where each row is [angle_start, a, b, c, d].
        """
        from scipy.interpolate import CubicSpline

        if periodic:
            # Ensure endpoints match for periodic spline
            if abs(self._radii[0] - self._radii[-1]) > 1e-10:
                # Adjust last point to match first
                radii = self._radii.copy()
                radii[-1] = radii[0]
            else:
                radii = self._radii
            spline = CubicSpline(self._angles, radii, bc_type="periodic")
        else:
            spline = CubicSpline(self._angles, self._radii, bc_type="natural")

        # Extract polynomial coefficients for each segment
        # CubicSpline.c has shape (4, n-1) with coefficients in descending power
        n_segments = len(self._angles) - 1
        coeffs = np.zeros((n_segments, 5), dtype=np.float64)

        for i in range(n_segments):
            # Store angle at start of segment
            coeffs[i, 0] = self._angles[i]

            # scipy stores c[k] for (x - x[i])^(3-k)
            # We want a + b*t + c*t^2 + d*t^3 where t = (x - x[i]) / h
            h = self._angles[i + 1] - self._angles[i]

            # c[0] is x^3 coeff, c[1] is x^2, c[2] is x, c[3] is constant
            # Convert from (x - x[i])^k form to normalized t form
            c = spline.c[:, i]

            # a = constant term = c[3]
            # For t-form with t = (x-xi)/h:
            # Original: c3 + c2*(x-xi) + c1*(x-xi)^2 + c0*(x-xi)^3
            # In t: c3 + c2*h*t + c1*h^2*t^2 + c0*h^3*t^3
            coeffs[i, 1] = c[3]  # a
            coeffs[i, 2] = c[2] * h  # b
            coeffs[i, 3] = c[1] * h**2  # c
            coeffs[i, 4] = c[0] * h**3  # d

        return coeffs

    def evaluate(self, angle: float) -> float:
        """Evaluate cam radius at given angle using spline interpolation."""
        return float(
            evaluate_spline_profile_displacement(
                angle,
                self._spline_coeffs[:, 0],  # angles
                self._spline_coeffs[:, 1:],  # coeffs [a, b, c, d]
            )
        )

    def evaluate_derivative(self, angle: float) -> float:
        """Evaluate cam radius derivative at given angle."""
        return float(
            evaluate_spline_profile_derivative(
                angle,
                self._spline_coeffs[:, 0],
                self._spline_coeffs[:, 1:],
            )
        )

    def get_constraints(self) -> tuple[float, ...]:
        """Return optimizable constraints (the radius values)."""
        return tuple(self._radii)

    def set_constraints(self, *radii: float | None) -> None:
        """Set radius values and recompute spline.

        Args:
            *radii: New radius values (same length as original).
        """
        new_radii = []
        for i, r in enumerate(radii):
            if r is not None:
                new_radii.append(r)
            elif i < len(self._radii):
                new_radii.append(self._radii[i])

        if len(new_radii) == len(self._radii):
            self._radii = np.array(new_radii, dtype=np.float64)
            self.base_radius = float(np.min(self._radii))
            # Recompute spline with new radii
            self._spline_coeffs = self._compute_spline_coefficients(periodic=True)

    def to_numba_data(self) -> tuple[int, NDArray[np.float64]]:
        """Convert to numba-compatible representation."""
        return (PROFILE_SPLINE, self._spline_coeffs)

    @property
    def angles(self) -> NDArray[np.float64]:
        """Return the angle values."""
        return self._angles.copy()

    @property
    def radii(self) -> NDArray[np.float64]:
        """Return the radius values."""
        return self._radii.copy()
