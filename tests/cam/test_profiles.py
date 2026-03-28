"""Tests for cam profiles - targeting uncovered lines in profiles.py."""

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
)
from pylinkage.cam.profiles import FunctionProfile, PointArrayProfile


class TestFunctionProfileExtended:
    """Extended tests for FunctionProfile covering uncovered branches."""

    def test_default_motion_law(self) -> None:
        """Test that default motion law is Harmonic."""
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        assert isinstance(profile.motion_law, HarmonicMotionLaw)

    def test_default_timing_parameters(self) -> None:
        """Test default timing when not specified."""
        profile = FunctionProfile()
        assert profile.rise_start == 0.0
        assert profile.rise_end == pytest.approx(math.pi / 2)
        assert profile.dwell_high_end == pytest.approx(math.pi)
        assert profile.fall_end == pytest.approx(3 * math.pi / 2)

    def test_custom_name(self) -> None:
        profile = FunctionProfile(name="MyProfile")
        assert profile.name == "MyProfile"

    def test_auto_name(self) -> None:
        profile = FunctionProfile()
        assert "FunctionProfile" in profile.name

    def test_evaluate_fall_phase(self) -> None:
        """Test evaluation during fall phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi / 2,
            dwell_high_end=math.pi,
            fall_end=3 * math.pi / 2,
        )
        # During fall phase: between dwell_high_end and fall_end
        r = profile.evaluate(5 * math.pi / 4)
        assert 1.0 < r < 1.5

    def test_evaluate_after_fall(self) -> None:
        """Test evaluation after fall (dwell-low)."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi / 2,
            dwell_high_end=math.pi,
            fall_end=3 * math.pi / 2,
        )
        # After fall_end, should be at base radius
        r = profile.evaluate(7 * math.pi / 4)
        assert abs(r - 1.0) < 1e-10

    def test_derivative_during_rise(self) -> None:
        """Test derivative during rise phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
        )
        # At midpoint of rise, derivative should be positive
        deriv = profile.evaluate_derivative(math.pi / 2)
        assert deriv > 0

    def test_derivative_during_fall(self) -> None:
        """Test derivative during fall phase."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi / 2,
            dwell_high_end=math.pi,
            fall_end=3 * math.pi / 2,
        )
        deriv = profile.evaluate_derivative(5 * math.pi / 4)
        assert deriv < 0

    def test_set_constraints_partial(self) -> None:
        """Test setting only some constraints."""
        profile = FunctionProfile(base_radius=1.0, total_lift=0.5)
        profile.set_constraints(2.0, None)
        assert profile.base_radius == 2.0
        assert profile.total_lift == 0.5

        profile.set_constraints(None, 0.75)
        assert profile.base_radius == 2.0
        assert profile.total_lift == 0.75

    def test_to_numba_data(self) -> None:
        """Test numba data conversion."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        profile_type, data = profile.to_numba_data()
        assert isinstance(profile_type, int)
        assert isinstance(data, np.ndarray)
        assert data[0] == 1.0  # base_radius
        assert data[1] == 0.5  # total_lift

    def test_to_numba_data_polynomial(self) -> None:
        """Test numba data with polynomial (which has coefficients)."""
        law = polynomial_345()
        profile = FunctionProfile(motion_law=law, base_radius=1.0, total_lift=0.5)
        profile_type, data = profile.to_numba_data()
        # Data should include timing params + polynomial coefficients
        assert len(data) > 6  # 6 timing params + coefficients

    def test_pressure_angle(self) -> None:
        """Test pressure angle computation."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
        )
        # During dwell, pressure angle should be zero (no derivative)
        pa = profile.pressure_angle(3 * math.pi / 2)
        assert abs(pa) < 1e-10

    def test_pressure_angle_during_motion(self) -> None:
        """Test pressure angle during rise."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
            rise_start=0.0,
            rise_end=math.pi,
        )
        pa = profile.pressure_angle(math.pi / 2)
        assert abs(pa) > 0

    def test_pitch_radius(self) -> None:
        """Test pitch radius computation with roller."""
        profile = FunctionProfile(
            motion_law=HarmonicMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        pr = profile.pitch_radius(0.0, roller_radius=0.1)
        assert isinstance(pr, float)
        assert pr > 0

    def test_with_cycloidal_law(self) -> None:
        """Test profile with cycloidal motion law."""
        profile = FunctionProfile(
            motion_law=CycloidalMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        assert abs(profile.evaluate(0.0) - 1.0) < 1e-10

    def test_with_modified_trapezoidal_law(self) -> None:
        """Test profile with modified trapezoidal motion law."""
        profile = FunctionProfile(
            motion_law=ModifiedTrapezoidalMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        assert abs(profile.evaluate(0.0) - 1.0) < 1e-10

    def test_with_polynomial_law(self) -> None:
        """Test profile with polynomial motion law."""
        profile = FunctionProfile(
            motion_law=PolynomialMotionLaw(),
            base_radius=1.0,
            total_lift=0.5,
        )
        assert abs(profile.evaluate(0.0) - 1.0) < 1e-10


class TestPointArrayProfileExtended:
    """Extended tests for PointArrayProfile."""

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            PointArrayProfile(angles=[0, 1, 2], radii=[1.0, 2.0])

    def test_derivative_at_knot(self) -> None:
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        deriv = profile.evaluate_derivative(math.pi / 2)
        assert isinstance(deriv, float)

    def test_derivative_between_knots(self) -> None:
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        deriv = profile.evaluate_derivative(math.pi / 4)
        assert isinstance(deriv, float)

    def test_non_periodic(self) -> None:
        """Test non-periodic spline (natural boundary conditions)."""
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.5, 2.0, 1.5, 1.2]
        profile = PointArrayProfile(angles=angles, radii=radii, periodic=False)
        r = profile.evaluate(math.pi)
        assert abs(r - 2.0) < 1e-6

    def test_periodic_mismatch_radii(self) -> None:
        """Test periodic profile when endpoints don't match (auto-adjusted)."""
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.2]  # Endpoints don't match
        profile = PointArrayProfile(angles=angles, radii=radii, periodic=True)
        # Should adjust last point to match first
        r_start = profile.evaluate(0.0)
        assert isinstance(r_start, float)

    def test_custom_name(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii, name="test_profile")
        assert profile.name == "test_profile"

    def test_auto_name(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        assert "PointArrayProfile" in profile.name

    def test_get_constraints(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        constraints = profile.get_constraints()
        assert constraints == (1.0, 1.5, 1.0)

    def test_set_constraints(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        profile.set_constraints(1.1, 1.6, 1.1)
        np.testing.assert_array_almost_equal(profile.radii, [1.1, 1.6, 1.1])
        assert profile.base_radius == pytest.approx(1.1)

    def test_set_constraints_partial_none(self) -> None:
        """Test set_constraints with None values (keep existing)."""
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        profile.set_constraints(None, 2.0, None)
        np.testing.assert_array_almost_equal(profile.radii, [1.0, 2.0, 1.0])

    def test_set_constraints_wrong_length(self) -> None:
        """Test that wrong number of constraints doesn't update."""
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        # Too few constraints - should not update
        profile.set_constraints(1.1)
        # Radii should be unchanged since length doesn't match
        np.testing.assert_array_almost_equal(profile.radii, [1.0, 1.5, 1.0])

    def test_to_numba_data(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        profile_type, data = profile.to_numba_data()
        assert isinstance(profile_type, int)
        assert isinstance(data, np.ndarray)

    def test_angles_property(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        a = profile.angles
        np.testing.assert_array_almost_equal(a, angles)

    def test_radii_property(self) -> None:
        angles = [0, math.pi, 2 * math.pi]
        radii = [1.0, 1.5, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        r = profile.radii
        np.testing.assert_array_almost_equal(r, radii)

    def test_pitch_radius(self) -> None:
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        pr = profile.pitch_radius(math.pi / 4, roller_radius=0.1)
        assert isinstance(pr, float)

    def test_pressure_angle(self) -> None:
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        radii = [1.0, 1.25, 1.5, 1.25, 1.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        pa = profile.pressure_angle(math.pi / 4)
        assert isinstance(pa, float)

    def test_pressure_angle_zero_radius(self) -> None:
        """Test pressure angle when radius is near zero."""
        # This tests the abs(cam_radius) < 1e-10 branch
        angles = [0, math.pi, 2 * math.pi]
        radii = [0.0, 1.0, 0.0]
        profile = PointArrayProfile(angles=angles, radii=radii)
        pa = profile.pressure_angle(0.0)
        assert pa == 0.0
