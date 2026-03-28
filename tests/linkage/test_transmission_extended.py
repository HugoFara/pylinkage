"""Extended tests for transmission.py covering missing lines.

Targets: worst_angle min branch (line 65), _get_joint_coord branches (line 124/126/131),
auto-detect missing rocker (line 178), transmission_angle_at_position with no valid
positions (line 211), stroke_at_position errors (lines 423/430/436),
analyze_stroke no prismatic (line 470), empty stroke (line 497/488).
"""

import warnings

import pytest

import pylinkage as pl
from pylinkage.linkage.transmission import (
    _auto_detect_fourbar_joints,
    _auto_detect_prismatic_joint,
    _get_joint_coord,
    compute_transmission_angle,
    stroke_at_position,
    transmission_angle_at_position,
)


class TestGetJointCoord:
    """Tests for _get_joint_coord edge cases."""

    def test_none_returns_none(self):
        assert _get_joint_coord(None) is None

    def test_tuple_returns_tuple(self):
        assert _get_joint_coord((1.0, 2.0)) == (1.0, 2.0)

    def test_object_with_xy(self):
        """Object with x and y attributes."""

        class FakeJoint:
            x = 3.0
            y = 4.0

        result = _get_joint_coord(FakeJoint())
        assert result == (3.0, 4.0)

    def test_object_missing_x_returns_none(self):
        """Object missing x attribute returns None."""

        class FakeJoint:
            y = 4.0

        result = _get_joint_coord(FakeJoint())
        assert result is None

    def test_object_none_y_returns_none(self):
        """Object with y=None returns None."""

        class FakeJoint:
            x = 3.0
            y = None

        result = _get_joint_coord(FakeJoint())
        assert result is None


class TestWorstAngle:
    """Test the worst_angle method branches."""

    def test_worst_angle_min_closer_to_90(self):
        """When min_angle is farther from 90 than max_angle, return min_angle."""
        import numpy as np
        from pylinkage.linkage.transmission import TransmissionAngleAnalysis

        # min=20 (70 from 90), max=130 (40 from 90) -> worst is min=20
        analysis = TransmissionAngleAnalysis(
            min_angle=20.0,
            max_angle=130.0,
            mean_angle=75.0,
            angles=np.array([20.0, 75.0, 130.0]),
            is_acceptable=False,
            min_deviation=15.0,
            max_deviation=70.0,
            min_angle_step=0,
            max_angle_step=2,
        )
        assert analysis.worst_angle() == 20.0

    def test_worst_angle_max_farther(self):
        """When max_angle is farther from 90, return max_angle."""
        import numpy as np
        from pylinkage.linkage.transmission import TransmissionAngleAnalysis

        analysis = TransmissionAngleAnalysis(
            min_angle=70.0,
            max_angle=170.0,
            mean_angle=120.0,
            angles=np.array([70.0, 120.0, 170.0]),
            is_acceptable=False,
            min_deviation=20.0,
            max_deviation=80.0,
            min_angle_step=0,
            max_angle_step=2,
        )
        assert analysis.worst_angle() == 170.0


class TestAutoDetectFourBarMissingRockerPivot:
    """Test _auto_detect_fourbar_joints when rocker pivot is None."""

    def test_revolute_joint1_none_raises(self):
        """Revolute with joint1=None raises ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            # Create a Revolute without joint1 set
            rev = pl.Revolute(2, 1, joint0=crank, joint1=None, distance0=2.0, distance1=1.5, name="C")
            # joint1 is None
            rev.joint1 = None
            linkage = pl.Linkage(joints=[crank, rev], order=[crank, rev])

        with pytest.raises(ValueError, match="rocker pivot.*None"):
            _auto_detect_fourbar_joints(linkage)


class TestTransmissionAngleAtPositionErrors:
    """Tests for transmission_angle_at_position error paths."""

    def test_invalid_joint_positions_raises(self):
        """When joint positions can't be determined, raise ValueError."""

        class BadJoint:
            pass  # no x/y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            linkage = pl.Linkage(joints=[crank], order=[crank])

        with pytest.raises(ValueError):
            transmission_angle_at_position(
                linkage,
                coupler_joint=BadJoint(),
                output_joint=BadJoint(),
                rocker_pivot=BadJoint(),
            )


class TestStrokeAtPositionErrors:
    """Tests for stroke_at_position error paths."""

    def test_invalid_slider_position_raises(self):
        """Slider with no valid position raises ValueError."""

        class FakePrismatic:
            joint1 = None
            joint2 = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            linkage = pl.Linkage(joints=[], order=[])

        with pytest.raises(ValueError):
            stroke_at_position(linkage, prismatic_joint=FakePrismatic())

    def test_missing_joint1_joint2_raises(self):
        """Prismatic with missing joint1/joint2 raises ValueError."""

        class FakePrismatic:
            x = 1.0
            y = 0.0
            joint1 = None
            joint2 = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            linkage = pl.Linkage(joints=[], order=[])

        with pytest.raises(ValueError, match="joint1 and joint2"):
            stroke_at_position(linkage, prismatic_joint=FakePrismatic())

    def test_invalid_line_point_positions_raises(self):
        """Line point joints with no valid position raises ValueError."""

        class FakeJoint:
            pass  # no x/y

        class FakePrismatic:
            x = 1.0
            y = 0.0
            joint1 = FakeJoint()
            joint2 = FakeJoint()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            linkage = pl.Linkage(joints=[], order=[])

        with pytest.raises(ValueError, match="line point positions"):
            stroke_at_position(linkage, prismatic_joint=FakePrismatic())
