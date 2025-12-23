"""Tests for transmission angle analysis."""

import math
import warnings

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.linkage.transmission import (
    TransmissionAngleAnalysis,
    analyze_transmission,
    compute_transmission_angle,
)


class TestComputeTransmissionAngle:
    """Test the core angle computation function."""

    def test_right_angle(self):
        """When links are perpendicular, angle should be 90 degrees."""
        # B at origin, C at (1, 0), D at (1, 1)
        # BC is horizontal right, DC is vertical down
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(1, 1),  # D
        )
        assert abs(angle - 90.0) < 1e-5

    def test_collinear_links_180(self):
        """When links point in opposite directions, angle is 180."""
        # B--C--D collinear
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(2, 0),  # D - beyond C
        )
        assert abs(angle - 180.0) < 1e-5

    def test_acute_angle_30_degrees(self):
        """Test 30 degree configuration."""
        # BC = (1, 0), need DC at 30 degrees
        # cos(30) = sqrt(3)/2 ≈ 0.866
        # DC = (sqrt(3)/2, 0.5) makes 30 degree angle with (1, 0)
        sqrt3_2 = math.sqrt(3) / 2
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(1 - sqrt3_2, -0.5),  # D: DC = (sqrt(3)/2, 0.5)
        )
        assert abs(angle - 30.0) < 1e-5

    def test_45_degree_angle(self):
        """Test 45 degree configuration."""
        # For 45 degrees, we need vectors at 45 degrees
        # BC = (1, 0), need DC such that angle is 45
        # DC = (1, 1) normalized: cos(45) = 1/sqrt(2)
        # So D at position where C - D = (1, 1) -> D = C - (1, 1) = (0, -1)
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(0, -1),  # D: DC = (1, 1)
        )
        assert abs(angle - 45.0) < 1e-5

    def test_135_degree_angle(self):
        """Test 135 degree configuration."""
        # BC = (1, 0), need DC at 135 degrees
        # cos(135) = -1/sqrt(2), so DC = (-1, 1) or (-1, -1)
        # D = C - (-1, 1) = (2, -1)
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(2, -1),  # D: DC = (-1, 1)
        )
        assert abs(angle - 135.0) < 1e-5

    def test_60_degree_angle(self):
        """Test 60 degree configuration."""
        # BC = (1, 0), need DC at 60 degrees
        # cos(60) = 0.5, DC should be (0.5, sqrt(3)/2) normalized to unit
        # For unit vector at 60 degrees: (cos(60), sin(60)) = (0.5, sqrt(3)/2)
        # D = C - DC = (1, 0) - (0.5, sqrt(3)/2) = (0.5, -sqrt(3)/2)
        sqrt3_2 = math.sqrt(3) / 2
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),  # B
            output_joint=(1, 0),  # C
            rocker_pivot=(0.5, -sqrt3_2),  # D: DC = (0.5, sqrt(3)/2)
        )
        assert abs(angle - 60.0) < 1e-5

    def test_degenerate_zero_length_coupler(self):
        """Handle zero-length coupler vector gracefully."""
        angle = compute_transmission_angle(
            coupler_joint=(1, 0),  # Same as output
            output_joint=(1, 0),
            rocker_pivot=(2, 0),
        )
        # Should return 90 for degenerate case
        assert angle == 90.0

    def test_degenerate_zero_length_rocker(self):
        """Handle zero-length rocker vector gracefully."""
        angle = compute_transmission_angle(
            coupler_joint=(0, 0),
            output_joint=(1, 0),
            rocker_pivot=(1, 0),  # Same as output
        )
        assert angle == 90.0


class TestFourBarTransmissionAnalysis:
    """Test transmission analysis on four-bar linkages."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="B",
            )
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
                name="C",
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
                name="Test Four-Bar",
            )
        return linkage

    def test_analyze_returns_dataclass(self, fourbar_linkage):
        """Check that analyze_transmission returns proper dataclass."""
        result = analyze_transmission(fourbar_linkage, iterations=10)
        assert isinstance(result, TransmissionAngleAnalysis)
        assert isinstance(result.min_angle, float)
        assert isinstance(result.max_angle, float)
        assert isinstance(result.mean_angle, float)
        assert isinstance(result.angles, np.ndarray)

    def test_angles_in_valid_range(self, fourbar_linkage):
        """All angles should be between 0 and 180 degrees."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        assert all(0 <= a <= 180 for a in result.angles)

    def test_min_max_consistency(self, fourbar_linkage):
        """min_angle should be <= mean_angle <= max_angle."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        assert result.min_angle <= result.max_angle
        assert result.min_angle <= result.mean_angle
        assert result.mean_angle <= result.max_angle

    def test_is_acceptable_logic(self, fourbar_linkage):
        """is_acceptable should match acceptable_range check."""
        result = analyze_transmission(
            fourbar_linkage,
            iterations=50,
            acceptable_range=(40.0, 140.0),
        )
        expected = result.min_angle >= 40.0 and result.max_angle <= 140.0
        assert result.is_acceptable == expected

    def test_deviation_from_optimal(self, fourbar_linkage):
        """max_deviation should be max(|angle - 90|)."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        calculated_max_dev = max(abs(a - 90.0) for a in result.angles)
        assert abs(result.max_deviation - calculated_max_dev) < 1e-5

    def test_min_deviation_from_optimal(self, fourbar_linkage):
        """min_deviation should be min(|angle - 90|)."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        calculated_min_dev = min(abs(a - 90.0) for a in result.angles)
        assert abs(result.min_deviation - calculated_min_dev) < 1e-5

    def test_step_indices(self, fourbar_linkage):
        """Step indices should point to actual min/max angles."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        assert result.angles[result.min_angle_step] == result.min_angle
        assert result.angles[result.max_angle_step] == result.max_angle

    def test_worst_angle(self, fourbar_linkage):
        """worst_angle should return angle with max deviation from 90."""
        result = analyze_transmission(fourbar_linkage, iterations=50)
        worst = result.worst_angle()
        assert worst in (result.min_angle, result.max_angle)
        # It should be the one further from 90
        if abs(result.max_angle - 90.0) >= abs(result.min_angle - 90.0):
            assert worst == result.max_angle
        else:
            assert worst == result.min_angle


class TestLinkageConvenienceMethods:
    """Test the convenience methods on Linkage class."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="B",
            )
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
                name="C",
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_transmission_angle_method(self, fourbar_linkage):
        """Test linkage.transmission_angle() returns float."""
        angle = fourbar_linkage.transmission_angle()
        assert isinstance(angle, float)
        assert 0 <= angle <= 180

    def test_analyze_transmission_method(self, fourbar_linkage):
        """Test linkage.analyze_transmission() returns analysis."""
        result = fourbar_linkage.analyze_transmission(iterations=10)
        assert isinstance(result, TransmissionAngleAnalysis)

    def test_analyze_transmission_custom_range(self, fourbar_linkage):
        """Test custom acceptable range."""
        result = fourbar_linkage.analyze_transmission(
            iterations=10,
            acceptable_range=(30.0, 150.0),
        )
        expected = result.min_angle >= 30.0 and result.max_angle <= 150.0
        assert result.is_acceptable == expected


class TestAutoDetection:
    """Test automatic joint detection."""

    def test_missing_crank_raises(self):
        """Should raise if no Crank joint found."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Linkage with only Static joints
            static = pl.Static(0, 0, name="A")
            linkage = pl.Linkage(joints=[static], order=[static])

        with pytest.raises(ValueError, match="no Crank"):
            analyze_transmission(linkage)

    def test_missing_revolute_raises(self):
        """Should raise if no Revolute joint found."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            linkage = pl.Linkage(joints=[crank], order=[crank])

        with pytest.raises(ValueError, match="no Revolute"):
            analyze_transmission(linkage)


class TestTransmissionAngleAnalysisDataclass:
    """Test the TransmissionAngleAnalysis dataclass."""

    def test_acceptable_range_property(self):
        """acceptable_range property should return (40, 140)."""
        analysis = TransmissionAngleAnalysis(
            min_angle=50.0,
            max_angle=130.0,
            mean_angle=90.0,
            angles=np.array([50.0, 90.0, 130.0]),
            is_acceptable=True,
            min_deviation=0.0,
            max_deviation=40.0,
            min_angle_step=0,
            max_angle_step=2,
        )
        assert analysis.acceptable_range == (40.0, 140.0)

    def test_frozen_dataclass(self):
        """Dataclass should be immutable."""
        analysis = TransmissionAngleAnalysis(
            min_angle=50.0,
            max_angle=130.0,
            mean_angle=90.0,
            angles=np.array([50.0, 90.0, 130.0]),
            is_acceptable=True,
            min_deviation=0.0,
            max_deviation=40.0,
            min_angle_step=0,
            max_angle_step=2,
        )
        with pytest.raises(AttributeError):
            analysis.min_angle = 60.0  # type: ignore[misc]
