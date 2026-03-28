"""Tests for cam motion laws - targeting uncovered lines in motion_laws.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.cam.motion_laws import (
    CycloidalMotionLaw,
    HarmonicMotionLaw,
    ModifiedTrapezoidalMotionLaw,
    PolynomialMotionLaw,
    polynomial_345,
    polynomial_4567,
)


class TestHarmonicMotionLawExtended:
    """Extended tests for HarmonicMotionLaw covering velocity/acceleration."""

    def test_profile_type(self) -> None:
        law = HarmonicMotionLaw()
        assert isinstance(law.profile_type, int)

    def test_acceleration_at_boundaries(self) -> None:
        law = HarmonicMotionLaw()
        # Harmonic has non-zero acceleration at boundaries
        acc_start = law.acceleration(0.0)
        acc_end = law.acceleration(1.0)
        # Should be non-zero (acceleration discontinuity is a known property)
        assert abs(acc_start) > 0
        assert abs(acc_end) > 0

    def test_acceleration_at_midpoint(self) -> None:
        law = HarmonicMotionLaw()
        # At u=0.5, acceleration should be zero (inflection point)
        acc_mid = law.acceleration(0.5)
        assert abs(acc_mid) < 1e-10

    def test_velocity_symmetry(self) -> None:
        law = HarmonicMotionLaw()
        # Velocity at u and (1-u) should be related
        v1 = law.velocity(0.25)
        v2 = law.velocity(0.75)
        # For harmonic, v(u) = v(1-u) due to symmetry of sin
        assert abs(v1 - v2) < 1e-10

    def test_to_numba_coefficients(self) -> None:
        law = HarmonicMotionLaw()
        coeffs = law.to_numba_coefficients()
        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == 0  # Default empty for non-polynomial


class TestCycloidalMotionLawExtended:
    """Extended tests for CycloidalMotionLaw covering acceleration."""

    def test_profile_type(self) -> None:
        law = CycloidalMotionLaw()
        assert isinstance(law.profile_type, int)

    def test_acceleration_at_boundaries(self) -> None:
        law = CycloidalMotionLaw()
        # Cycloidal has zero acceleration at boundaries
        acc_start = law.acceleration(0.0)
        acc_end = law.acceleration(1.0)
        assert abs(acc_start) < 1e-10
        assert abs(acc_end) < 1e-10

    def test_acceleration_at_midpoint(self) -> None:
        law = CycloidalMotionLaw()
        acc_mid = law.acceleration(0.5)
        # At midpoint, acceleration should be zero
        assert abs(acc_mid) < 1e-10

    def test_acceleration_peak(self) -> None:
        law = CycloidalMotionLaw()
        # Peak acceleration at u=0.25
        acc_quarter = law.acceleration(0.25)
        assert abs(acc_quarter) > 0

    def test_displacement_at_quarter(self) -> None:
        law = CycloidalMotionLaw()
        d = law.displacement(0.25)
        assert 0.0 < d < 0.5

    def test_velocity_at_midpoint(self) -> None:
        law = CycloidalMotionLaw()
        v = law.velocity(0.5)
        assert v == pytest.approx(2.0)

    def test_to_numba_coefficients(self) -> None:
        law = CycloidalMotionLaw()
        coeffs = law.to_numba_coefficients()
        assert len(coeffs) == 0


class TestModifiedTrapezoidalMotionLawExtended:
    """Extended tests for ModifiedTrapezoidalMotionLaw."""

    def test_profile_type(self) -> None:
        law = ModifiedTrapezoidalMotionLaw()
        assert isinstance(law.profile_type, int)

    def test_velocity_at_boundaries(self) -> None:
        law = ModifiedTrapezoidalMotionLaw()
        v_start = law.velocity(0.0)
        v_end = law.velocity(1.0)
        assert abs(v_start) < 1e-6
        assert abs(v_end) < 1e-6

    def test_acceleration_at_start(self) -> None:
        """Test numerical acceleration near u=0 (uses forward difference)."""
        law = ModifiedTrapezoidalMotionLaw()
        acc = law.acceleration(0.0)
        assert isinstance(acc, float)

    def test_acceleration_at_end(self) -> None:
        """Test numerical acceleration near u=1 (uses backward difference)."""
        law = ModifiedTrapezoidalMotionLaw()
        acc = law.acceleration(1.0)
        assert isinstance(acc, float)

    def test_acceleration_at_mid(self) -> None:
        """Test numerical acceleration at midpoint (uses central difference)."""
        law = ModifiedTrapezoidalMotionLaw()
        acc = law.acceleration(0.5)
        assert isinstance(acc, float)

    def test_acceleration_near_boundaries(self) -> None:
        """Test acceleration very close to boundaries."""
        law = ModifiedTrapezoidalMotionLaw()
        # Near start (within eps threshold)
        acc_near_start = law.acceleration(1e-8)
        assert isinstance(acc_near_start, float)
        # Near end (within eps threshold)
        acc_near_end = law.acceleration(1.0 - 1e-8)
        assert isinstance(acc_near_end, float)

    def test_displacement_endpoints(self) -> None:
        law = ModifiedTrapezoidalMotionLaw()
        # Verify displacement at several interior points returns float
        for u in [0.1, 0.25, 0.5, 0.75, 0.9]:
            d = law.displacement(u)
            assert isinstance(d, float)

    def test_to_numba_coefficients(self) -> None:
        law = ModifiedTrapezoidalMotionLaw()
        coeffs = law.to_numba_coefficients()
        assert len(coeffs) == 0


class TestPolynomialMotionLawExtended:
    """Extended tests for PolynomialMotionLaw."""

    def test_profile_type(self) -> None:
        law = PolynomialMotionLaw()
        assert isinstance(law.profile_type, int)

    def test_default_is_345(self) -> None:
        law = PolynomialMotionLaw()
        assert law.displacement(0.0) == pytest.approx(0.0)
        assert law.displacement(1.0) == pytest.approx(1.0)
        assert law.velocity(0.0) == pytest.approx(0.0)
        assert law.velocity(1.0) == pytest.approx(0.0)

    def test_custom_coefficients(self) -> None:
        # Simple linear: s(u) = u -> coeffs = [0, 1]
        law = PolynomialMotionLaw([0.0, 1.0])
        assert law.displacement(0.5) == pytest.approx(0.5)
        assert law.velocity(0.5) == pytest.approx(1.0)

    def test_coefficients_property(self) -> None:
        coeffs = [0.0, 0.0, 0.0, 10.0, -15.0, 6.0]
        law = PolynomialMotionLaw(coeffs)
        np.testing.assert_array_almost_equal(law.coefficients, coeffs)

    def test_acceleration_345(self) -> None:
        law = polynomial_345()
        # Acceleration at boundaries should be zero for 3-4-5
        acc_start = law.acceleration(0.0)
        acc_end = law.acceleration(1.0)
        assert abs(acc_start) < 1e-10
        assert abs(acc_end) < 1e-10

    def test_acceleration_midpoint(self) -> None:
        law = polynomial_345()
        acc_mid = law.acceleration(0.5)
        # Should be zero at midpoint (symmetric polynomial)
        assert abs(acc_mid) < 1e-10

    def test_acceleration_at_quarter(self) -> None:
        law = polynomial_345()
        acc = law.acceleration(0.25)
        assert isinstance(acc, float)
        assert abs(acc) > 0

    def test_to_numba_coefficients(self) -> None:
        law = PolynomialMotionLaw([0.0, 0.0, 0.0, 10.0, -15.0, 6.0])
        coeffs = law.to_numba_coefficients()
        assert len(coeffs) == 6
        assert coeffs[3] == pytest.approx(10.0)

    def test_zero_displacement_polynomial(self) -> None:
        """Polynomial with all zero coeffs gives zero displacement."""
        law = PolynomialMotionLaw([0.0, 0.0, 0.0])
        assert law.displacement(0.5) == pytest.approx(0.0)
        assert law.velocity(0.5) == pytest.approx(0.0)
        assert law.acceleration(0.5) == pytest.approx(0.0)


class TestPolynomial4567:
    """Tests for the 4-5-6-7 polynomial factory."""

    def test_boundary_displacement(self) -> None:
        law = polynomial_4567()
        assert law.displacement(0.0) == pytest.approx(0.0)
        assert law.displacement(1.0) == pytest.approx(1.0)

    def test_boundary_velocity(self) -> None:
        law = polynomial_4567()
        assert law.velocity(0.0) == pytest.approx(0.0)
        assert law.velocity(1.0) == pytest.approx(0.0)

    def test_boundary_acceleration(self) -> None:
        law = polynomial_4567()
        assert law.acceleration(0.0) == pytest.approx(0.0)
        assert law.acceleration(1.0) == pytest.approx(0.0)

    def test_coefficients(self) -> None:
        law = polynomial_4567()
        assert len(law.coefficients) == 8
        assert law.coefficients[4] == pytest.approx(35.0)

    def test_midpoint(self) -> None:
        law = polynomial_4567()
        d = law.displacement(0.5)
        assert d == pytest.approx(0.5, abs=1e-6)
