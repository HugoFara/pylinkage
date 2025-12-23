"""Standard cam motion laws.

Motion laws define the dimensionless displacement function s(u)
where u in [0, 1] is the normalized angle and s in [0, 1] is
the normalized displacement.

Available motion laws:
    HarmonicMotionLaw: Simple harmonic (cosine) motion
    CycloidalMotionLaw: Cycloidal motion (zero velocity/acceleration at ends)
    ModifiedTrapezoidalMotionLaw: Modified trapezoidal (low peak acceleration)
    PolynomialMotionLaw: General polynomial (3-4-5 or custom)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ._numba_core import (
    PROFILE_CYCLOIDAL,
    PROFILE_HARMONIC,
    PROFILE_MODIFIED_TRAPEZOIDAL,
    PROFILE_POLYNOMIAL,
    evaluate_cycloidal,
    evaluate_cycloidal_acceleration,
    evaluate_cycloidal_velocity,
    evaluate_harmonic,
    evaluate_harmonic_acceleration,
    evaluate_harmonic_velocity,
    evaluate_modified_trapezoidal,
    evaluate_modified_trapezoidal_velocity,
    evaluate_polynomial,
    evaluate_polynomial_velocity,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MotionLaw(ABC):
    """Abstract base class for cam motion laws.

    Motion laws define the dimensionless displacement function s(u)
    where u in [0, 1] is the normalized angle and s in [0, 1] is
    the normalized displacement.

    Subclasses must implement displacement(), velocity(), and acceleration()
    methods that operate on normalized coordinates.
    """

    @property
    @abstractmethod
    def profile_type(self) -> int:
        """Return the profile type code for numba dispatch."""
        ...

    @abstractmethod
    def displacement(self, u: float) -> float:
        """Compute dimensionless displacement s(u).

        Args:
            u: Normalized angle in [0, 1].

        Returns:
            Normalized displacement in [0, 1].
        """
        ...

    @abstractmethod
    def velocity(self, u: float) -> float:
        """Compute dimensionless velocity ds/du.

        Args:
            u: Normalized angle in [0, 1].

        Returns:
            Normalized velocity.
        """
        ...

    @abstractmethod
    def acceleration(self, u: float) -> float:
        """Compute dimensionless acceleration d2s/du2.

        Args:
            u: Normalized angle in [0, 1].

        Returns:
            Normalized acceleration.
        """
        ...

    def to_numba_coefficients(self) -> NDArray[np.float64]:
        """Return coefficients for numba evaluation.

        Default implementation returns empty array.
        Override for polynomial/custom motion laws.

        Returns:
            Numpy array of coefficients.
        """
        return np.array([], dtype=np.float64)


class HarmonicMotionLaw(MotionLaw):
    """Simple harmonic (cosine) motion law.

    s(u) = (1 - cos(pi * u)) / 2

    Properties:
        - Smooth displacement curve
        - Non-zero acceleration at boundaries (acceleration discontinuity)
        - Maximum velocity at u = 0.5
        - Simple and widely used

    Example:
        >>> law = HarmonicMotionLaw()
        >>> law.displacement(0.0)  # Start
        0.0
        >>> law.displacement(0.5)  # Midpoint
        0.5
        >>> law.displacement(1.0)  # End
        1.0
    """

    @property
    def profile_type(self) -> int:
        """Return harmonic profile type code."""
        return PROFILE_HARMONIC

    def displacement(self, u: float) -> float:
        """Compute harmonic displacement."""
        return float(evaluate_harmonic(u))

    def velocity(self, u: float) -> float:
        """Compute harmonic velocity."""
        return float(evaluate_harmonic_velocity(u))

    def acceleration(self, u: float) -> float:
        """Compute harmonic acceleration."""
        return float(evaluate_harmonic_acceleration(u))


class CycloidalMotionLaw(MotionLaw):
    """Cycloidal motion law.

    s(u) = u - sin(2*pi*u) / (2*pi)

    Properties:
        - Zero velocity at boundaries (smooth start/stop)
        - Zero acceleration at boundaries (no jerk discontinuity)
        - Higher peak acceleration than harmonic
        - Best for high-speed applications requiring smooth motion

    Example:
        >>> law = CycloidalMotionLaw()
        >>> law.displacement(0.0)  # Zero displacement at start
        0.0
        >>> law.velocity(0.0)  # Zero velocity at start
        0.0
        >>> law.velocity(0.5)  # Maximum velocity at midpoint
        2.0
    """

    @property
    def profile_type(self) -> int:
        """Return cycloidal profile type code."""
        return PROFILE_CYCLOIDAL

    def displacement(self, u: float) -> float:
        """Compute cycloidal displacement."""
        return float(evaluate_cycloidal(u))

    def velocity(self, u: float) -> float:
        """Compute cycloidal velocity."""
        return float(evaluate_cycloidal_velocity(u))

    def acceleration(self, u: float) -> float:
        """Compute cycloidal acceleration."""
        return float(evaluate_cycloidal_acceleration(u))


class ModifiedTrapezoidalMotionLaw(MotionLaw):
    """Modified trapezoidal motion law.

    Uses sinusoidal acceleration segments at start/end with
    constant acceleration in the middle.

    Properties:
        - Lower peak acceleration than harmonic or cycloidal
        - Zero velocity at boundaries
        - Continuous acceleration (no jerk discontinuity)
        - Good for high-load applications

    Example:
        >>> law = ModifiedTrapezoidalMotionLaw()
        >>> law.displacement(0.0)
        0.0
        >>> law.displacement(1.0)
        1.0
    """

    @property
    def profile_type(self) -> int:
        """Return modified trapezoidal profile type code."""
        return PROFILE_MODIFIED_TRAPEZOIDAL

    def displacement(self, u: float) -> float:
        """Compute modified trapezoidal displacement."""
        return float(evaluate_modified_trapezoidal(u))

    def velocity(self, u: float) -> float:
        """Compute modified trapezoidal velocity."""
        return float(evaluate_modified_trapezoidal_velocity(u))

    def acceleration(self, u: float) -> float:
        """Compute modified trapezoidal acceleration (numerical)."""
        # Use finite difference for acceleration
        eps = 1e-6
        if u < eps:
            return (self.velocity(eps) - self.velocity(0.0)) / eps
        elif u > 1.0 - eps:
            return (self.velocity(1.0) - self.velocity(1.0 - eps)) / eps
        else:
            return (self.velocity(u + eps) - self.velocity(u - eps)) / (2 * eps)


class PolynomialMotionLaw(MotionLaw):
    """General polynomial motion law.

    s(u) = sum(coefficients[i] * u^i)

    The polynomial must satisfy boundary conditions:
        - s(0) = 0, s(1) = 1 (displacement)
        - ds/du(0) = 0, ds/du(1) = 0 (velocity, typically)

    Common polynomials:
        - 3-4-5 polynomial: [0, 0, 0, 10, -15, 6]
        - 4-5-6-7 polynomial: [0, 0, 0, 0, 35, -84, 70, -20]

    Example:
        >>> # 3-4-5 polynomial (zero velocity and acceleration at ends)
        >>> law = PolynomialMotionLaw([0, 0, 0, 10, -15, 6])
        >>> law.displacement(0.0)
        0.0
        >>> law.displacement(1.0)
        1.0
        >>> law.velocity(0.0)
        0.0
    """

    def __init__(self, coefficients: list[float] | None = None) -> None:
        """Create a polynomial motion law.

        Args:
            coefficients: Polynomial coefficients [a0, a1, a2, ...].
                         Default is 3-4-5 polynomial.
        """
        if coefficients is None:
            # Default: 3-4-5 polynomial
            # s(u) = 10*u^3 - 15*u^4 + 6*u^5
            coefficients = [0.0, 0.0, 0.0, 10.0, -15.0, 6.0]
        self._coefficients = np.array(coefficients, dtype=np.float64)

    @property
    def profile_type(self) -> int:
        """Return polynomial profile type code."""
        return PROFILE_POLYNOMIAL

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Return polynomial coefficients."""
        return self._coefficients

    def displacement(self, u: float) -> float:
        """Compute polynomial displacement."""
        return float(evaluate_polynomial(u, self._coefficients))

    def velocity(self, u: float) -> float:
        """Compute polynomial velocity."""
        return float(evaluate_polynomial_velocity(u, self._coefficients))

    def acceleration(self, u: float) -> float:
        """Compute polynomial acceleration."""
        # Second derivative: sum(i*(i-1)*coeffs[i]*u^(i-2))
        result = 0.0
        power = 1.0
        for i in range(2, len(self._coefficients)):
            result += i * (i - 1) * self._coefficients[i] * power
            power *= u
        return result

    def to_numba_coefficients(self) -> NDArray[np.float64]:
        """Return polynomial coefficients for numba evaluation."""
        return self._coefficients


def polynomial_345() -> PolynomialMotionLaw:
    """Create a 3-4-5 polynomial motion law.

    s(u) = 10*u^3 - 15*u^4 + 6*u^5

    This polynomial has:
        - Zero velocity at u=0 and u=1
        - Zero acceleration at u=0 and u=1

    Returns:
        PolynomialMotionLaw instance.
    """
    return PolynomialMotionLaw([0.0, 0.0, 0.0, 10.0, -15.0, 6.0])


def polynomial_4567() -> PolynomialMotionLaw:
    """Create a 4-5-6-7 polynomial motion law.

    s(u) = 35*u^4 - 84*u^5 + 70*u^6 - 20*u^7

    This polynomial has:
        - Zero velocity at u=0 and u=1
        - Zero acceleration at u=0 and u=1
        - Zero jerk at u=0 and u=1

    Returns:
        PolynomialMotionLaw instance.
    """
    return PolynomialMotionLaw([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0])
