"""Tests for kinematic analysis (transmission angle and stroke)."""

import math
import warnings

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.linkage.transmission import (
    StrokeAnalysis,
    TransmissionAngleAnalysis,
    analyze_stroke,
    analyze_transmission,
    compute_slide_position,
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
                1,
                0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="B",
            )
            revolute = pl.Revolute(
                3,
                1,
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
                1,
                0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="B",
            )
            revolute = pl.Revolute(
                3,
                1,
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


# =============================================================================
# Stroke Analysis Tests (for Prismatic joints)
# =============================================================================


class TestComputeSlidePosition:
    """Test the core slide position computation function."""

    def test_position_at_line_origin(self):
        """Slider at line origin should have position 0."""
        pos = compute_slide_position(
            slider_pos=(0, 0),
            line_point1=(0, 0),
            line_point2=(1, 0),
        )
        assert abs(pos - 0.0) < 1e-10

    def test_position_at_line_endpoint(self):
        """Slider at second line point should have position = line length."""
        pos = compute_slide_position(
            slider_pos=(3, 0),
            line_point1=(0, 0),
            line_point2=(3, 0),
        )
        assert abs(pos - 3.0) < 1e-10

    def test_position_midway(self):
        """Slider at midpoint should have position = half line length."""
        pos = compute_slide_position(
            slider_pos=(1.5, 0),
            line_point1=(0, 0),
            line_point2=(3, 0),
        )
        assert abs(pos - 1.5) < 1e-10

    def test_negative_position(self):
        """Slider before line origin should have negative position."""
        pos = compute_slide_position(
            slider_pos=(-2, 0),
            line_point1=(0, 0),
            line_point2=(1, 0),
        )
        assert abs(pos - (-2.0)) < 1e-10

    def test_diagonal_line(self):
        """Test with a diagonal line (45 degrees)."""
        # Line from (0,0) to (1,1), slider at (0.5, 0.5)
        # Distance along line = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ≈ 0.707
        pos = compute_slide_position(
            slider_pos=(0.5, 0.5),
            line_point1=(0, 0),
            line_point2=(1, 1),
        )
        expected = math.sqrt(0.5)
        assert abs(pos - expected) < 1e-10

    def test_vertical_line(self):
        """Test with a vertical line."""
        pos = compute_slide_position(
            slider_pos=(0, 5),
            line_point1=(0, 0),
            line_point2=(0, 1),
        )
        assert abs(pos - 5.0) < 1e-10

    def test_slider_off_line_projected(self):
        """Slider off the line should project onto it."""
        # Slider at (2, 3) on horizontal line from (0,0) to (1,0)
        # Should project to x=2
        pos = compute_slide_position(
            slider_pos=(2, 3),
            line_point1=(0, 0),
            line_point2=(1, 0),
        )
        assert abs(pos - 2.0) < 1e-10

    def test_degenerate_zero_length_line(self):
        """Handle zero-length line gracefully."""
        pos = compute_slide_position(
            slider_pos=(5, 5),
            line_point1=(0, 0),
            line_point2=(0, 0),  # Same as point1
        )
        assert pos == 0.0


class TestSliderCrankStrokeAnalysis:
    """Test stroke analysis on slider-crank linkages."""

    @pytest.fixture
    def slider_crank_linkage(self):
        """Create a slider-crank mechanism."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Crank rotating around origin
            crank = pl.Crank(
                1,
                0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="crank",
            )
            # Slider moving along horizontal line
            slider = pl.Prismatic(
                3,
                0,
                joint0=crank,  # Connected to crank
                joint1=(0, 0),  # Line point 1
                joint2=(10, 0),  # Line point 2 (horizontal line)
                revolute_radius=3.0,  # Connecting rod length
                name="slider",
            )
            linkage = pl.Linkage(
                joints=[crank, slider],
                order=[crank, slider],
                name="Slider-Crank",
            )
        return linkage

    def test_analyze_returns_dataclass(self, slider_crank_linkage):
        """Check that analyze_stroke returns proper dataclass."""
        result = analyze_stroke(slider_crank_linkage, iterations=10)
        assert isinstance(result, StrokeAnalysis)
        assert isinstance(result.min_position, float)
        assert isinstance(result.max_position, float)
        assert isinstance(result.mean_position, float)
        assert isinstance(result.stroke_range, float)
        assert isinstance(result.positions, np.ndarray)

    def test_stroke_range_positive(self, slider_crank_linkage):
        """Stroke range should be positive."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        assert result.stroke_range >= 0

    def test_min_max_consistency(self, slider_crank_linkage):
        """min_position should be <= mean_position <= max_position."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        assert result.min_position <= result.max_position
        assert result.min_position <= result.mean_position
        assert result.mean_position <= result.max_position

    def test_stroke_range_calculation(self, slider_crank_linkage):
        """stroke_range should equal max - min."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        expected_range = result.max_position - result.min_position
        assert abs(result.stroke_range - expected_range) < 1e-10

    def test_step_indices(self, slider_crank_linkage):
        """Step indices should point to actual min/max positions."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        assert result.positions[result.min_position_step] == result.min_position
        assert result.positions[result.max_position_step] == result.max_position

    def test_amplitude_property(self, slider_crank_linkage):
        """amplitude should be half the stroke range."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        assert abs(result.amplitude - result.stroke_range / 2.0) < 1e-10

    def test_center_position_property(self, slider_crank_linkage):
        """center_position should be midpoint of min and max."""
        result = analyze_stroke(slider_crank_linkage, iterations=50)
        expected = (result.min_position + result.max_position) / 2.0
        assert abs(result.center_position - expected) < 1e-10

    def test_slider_crank_expected_stroke(self, slider_crank_linkage):
        """For crank radius r=1 and rod length L=3, stroke should be 2*r=2."""
        result = analyze_stroke(slider_crank_linkage, iterations=100)
        # Theoretical stroke = 2 * crank_radius = 2 * 1 = 2
        # But actual stroke depends on geometry - should be close to 2
        assert 1.5 < result.stroke_range < 2.5


class TestLinkageStrokeConvenienceMethods:
    """Test the stroke convenience methods on Linkage class."""

    @pytest.fixture
    def slider_crank_linkage(self):
        """Create a slider-crank mechanism."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1,
                0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
            )
            slider = pl.Prismatic(
                3,
                0,
                joint0=crank,
                joint1=(0, 0),
                joint2=(10, 0),
                revolute_radius=3.0,
            )
            linkage = pl.Linkage(
                joints=[crank, slider],
                order=[crank, slider],
            )
        return linkage

    def test_stroke_position_method(self, slider_crank_linkage):
        """Test linkage.stroke_position() returns float."""
        pos = slider_crank_linkage.stroke_position()
        assert isinstance(pos, float)

    def test_analyze_stroke_method(self, slider_crank_linkage):
        """Test linkage.analyze_stroke() returns analysis."""
        result = slider_crank_linkage.analyze_stroke(iterations=10)
        assert isinstance(result, StrokeAnalysis)


class TestStrokeAutoDetection:
    """Test automatic prismatic joint detection."""

    def test_missing_prismatic_raises(self):
        """Should raise if no Prismatic joint found."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            linkage = pl.Linkage(joints=[crank], order=[crank])

        with pytest.raises(ValueError, match="no Prismatic"):
            analyze_stroke(linkage)


class TestStrokeAnalysisDataclass:
    """Test the StrokeAnalysis dataclass."""

    def test_amplitude_property(self):
        """amplitude should be half the stroke range."""
        analysis = StrokeAnalysis(
            min_position=2.0,
            max_position=6.0,
            mean_position=4.0,
            stroke_range=4.0,
            positions=np.array([2.0, 4.0, 6.0]),
            min_position_step=0,
            max_position_step=2,
        )
        assert analysis.amplitude == 2.0

    def test_center_position_property(self):
        """center_position should be midpoint."""
        analysis = StrokeAnalysis(
            min_position=2.0,
            max_position=6.0,
            mean_position=4.0,
            stroke_range=4.0,
            positions=np.array([2.0, 4.0, 6.0]),
            min_position_step=0,
            max_position_step=2,
        )
        assert analysis.center_position == 4.0

    def test_frozen_dataclass(self):
        """Dataclass should be immutable."""
        analysis = StrokeAnalysis(
            min_position=2.0,
            max_position=6.0,
            mean_position=4.0,
            stroke_range=4.0,
            positions=np.array([2.0, 4.0, 6.0]),
            min_position_step=0,
            max_position_step=2,
        )
        with pytest.raises(AttributeError):
            analysis.min_position = 1.0  # type: ignore[misc]


class TestTransmissionAnglePlot:
    """Tests for TransmissionAngleAnalysis.plot()."""

    def _make_analysis(self, n=20):
        # Sweep cleanly through the acceptable range so legend lines all show.
        return TransmissionAngleAnalysis(
            min_angle=70.0,
            max_angle=110.0,
            mean_angle=90.0,
            angles=np.linspace(70.0, 110.0, n),
            is_acceptable=True,
            min_deviation=0.0,
            max_deviation=20.0,
            min_angle_step=0,
            max_angle_step=n - 1,
        )

    def test_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.axes import Axes

        result = self._make_analysis()
        ax = result.plot()
        assert isinstance(ax, Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_uses_provided_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis()
        fig, ax = plt.subplots()
        out = result.plot(ax=ax)
        assert out is ax
        plt.close(fig)

    def test_x_axis_spans_full_revolution(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis(n=60)
        ax = result.plot()
        # The transmission curve is the first artist
        line = ax.get_lines()[0]
        xs = line.get_xdata()
        assert xs[0] == 0.0
        assert xs[-1] < 360.0  # endpoint=False
        assert xs[-1] > 350.0  # close to a full revolution
        plt.close("all")

    def test_y_axis_clamped(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis()
        ax = result.plot()
        assert ax.get_ylim() == (0.0, 180.0)
        plt.close("all")

    def test_show_limits_and_optimum_lines(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis()
        ax = result.plot()
        # 1 transmission curve + 2 limit lines + 1 optimum line = 4 lines
        assert len(ax.get_lines()) == 4
        plt.close("all")

    def test_can_disable_reference_lines(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis()
        ax = result.plot(show_limits=False, show_optimum=False)
        assert len(ax.get_lines()) == 1
        plt.close("all")

    def test_title_can_be_omitted(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._make_analysis()
        ax = result.plot(title=None)
        assert ax.get_title() == ""
        plt.close("all")
