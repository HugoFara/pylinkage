"""Tests for pylinkage.linkage.transmission.

Covers transmission angle analysis for RRR joints and stroke/slide
analysis for RRP (prismatic) joints, including public dataclasses,
auto-detection helpers, degenerate inputs, and matplotlib plotting.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRPDyad, RRRDyad
from pylinkage.linkage.transmission import (
    StrokeAnalysis,
    TransmissionAngleAnalysis,
    _auto_detect_fourbar_joints,
    _auto_detect_prismatic_joint,
    _get_joint_coord,
    analyze_stroke,
    analyze_transmission,
    compute_slide_position,
    compute_transmission_angle,
    stroke_at_position,
    transmission_angle_at_position,
)
from pylinkage.simulation import Linkage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fourbar() -> Linkage:
    o1 = Ground(0.0, 0.0, name="O1")
    o2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=o2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([o1, o2, crank, rocker], name="FourBar")


def _make_slider_crank() -> Linkage:
    """A simple crank-slider: crank pivoting at origin, slider moving along
    the x-axis through the ground-defined line (O2, O3)."""
    o1 = Ground(0.0, 0.0, name="O1")
    # Slider guide line: two ground points along x-axis at y=0.1 offset
    o2 = Ground(0.0, 0.1, name="O2")
    o3 = Ground(5.0, 0.1, name="O3")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=o2,
        line_anchor2=o3,
        distance=3.0,
        name="slider",
    )
    return Linkage([o1, o2, o3, crank, slider], name="SliderCrank")


# ---------------------------------------------------------------------------
# compute_transmission_angle
# ---------------------------------------------------------------------------


def test_compute_transmission_angle_right_angle():
    # B=(0,0), C=(1,0), D=(1,1): BC along +x, DC along -y -> 90 degrees
    angle = compute_transmission_angle((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert math.isclose(angle, 90.0, abs_tol=1e-9)


def test_compute_transmission_angle_parallel():
    # BC and DC colinear in the same direction -> 0 degrees
    angle = compute_transmission_angle((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0))
    assert math.isclose(angle, 0.0, abs_tol=1e-9)


def test_compute_transmission_angle_opposite():
    # B=(0,0), C=(1,0), D=(2,0): BC along +x, DC along -x -> 180 degrees
    angle = compute_transmission_angle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    assert math.isclose(angle, 180.0, abs_tol=1e-9)


def test_compute_transmission_angle_degenerate_zero_bc():
    # When mag_bc is zero, function returns 90.0 as a fallback.
    assert compute_transmission_angle((1.0, 1.0), (1.0, 1.0), (2.0, 2.0)) == 90.0


def test_compute_transmission_angle_degenerate_zero_dc():
    assert compute_transmission_angle((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)) == 90.0


def test_compute_transmission_angle_45_degrees():
    angle = compute_transmission_angle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    assert math.isclose(angle, 45.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# _get_joint_coord
# ---------------------------------------------------------------------------


def test_get_joint_coord_none_returns_none():
    assert _get_joint_coord(None) is None


def test_get_joint_coord_tuple_returns_tuple():
    assert _get_joint_coord((1.0, 2.0)) == (1.0, 2.0)


def test_get_joint_coord_from_object_with_xy():
    class J:
        x = 1.5
        y = 2.5

    assert _get_joint_coord(J()) == (1.5, 2.5)


def test_get_joint_coord_object_with_missing_attr():
    class J:
        x = None
        y = None

    assert _get_joint_coord(J()) is None


# ---------------------------------------------------------------------------
# _auto_detect_fourbar_joints
# ---------------------------------------------------------------------------


def test_auto_detect_fourbar_joints_finds_all_three():
    lk = _make_fourbar()
    crank, revolute, rocker_pivot = _auto_detect_fourbar_joints(lk)
    assert type(crank).__name__ == "Crank"
    assert type(revolute).__name__ == "RRRDyad"
    # rocker_pivot is the anchor2 of the RRR dyad (= O2)
    assert rocker_pivot is lk.components[1]


def test_auto_detect_fourbar_joints_missing_crank_raises():
    o1 = Ground(0.0, 0.0, name="O1")
    # no crank in the linkage - should raise
    lk = Linkage([o1], name="empty")
    with pytest.raises(ValueError, match="no Crank joint"):
        _auto_detect_fourbar_joints(lk)


def test_auto_detect_fourbar_joints_missing_dyad_raises():
    o1 = Ground(0.0, 0.0, name="O1")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    lk = Linkage([o1, crank], name="crankonly")
    with pytest.raises(ValueError, match="no Revolute joint"):
        _auto_detect_fourbar_joints(lk)


# ---------------------------------------------------------------------------
# transmission_angle_at_position
# ---------------------------------------------------------------------------


def test_transmission_angle_at_position_auto_detects():
    lk = _make_fourbar()
    # Run a single step so every joint has coordinates
    next(lk.step(iterations=1))
    angle = transmission_angle_at_position(lk)
    assert 0.0 <= angle <= 180.0


def test_transmission_angle_at_position_explicit_joints():
    lk = _make_fourbar()
    next(lk.step(iterations=1))
    angle = transmission_angle_at_position(
        lk,
        coupler_joint=lk.components[2],
        output_joint=lk.components[3],
        rocker_pivot=lk.components[1],
    )
    assert 0.0 <= angle <= 180.0


def test_transmission_angle_at_position_invalid_coords_raises():
    class BadJoint:
        x = None
        y = None

    class OK:
        x = 0.0
        y = 0.0

    with pytest.raises(ValueError, match="Could not determine"):
        transmission_angle_at_position(
            None,
            coupler_joint=BadJoint(),
            output_joint=OK(),
            rocker_pivot=OK(),
        )


# ---------------------------------------------------------------------------
# analyze_transmission + TransmissionAngleAnalysis
# ---------------------------------------------------------------------------


def test_analyze_transmission_returns_dataclass_with_expected_fields():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12)
    assert isinstance(result, TransmissionAngleAnalysis)
    assert 0.0 <= result.min_angle <= 180.0
    assert 0.0 <= result.max_angle <= 180.0
    assert result.min_angle <= result.mean_angle <= result.max_angle
    assert result.angles.size == 12
    assert isinstance(result.is_acceptable, bool)
    assert result.min_deviation >= 0.0
    assert result.max_deviation >= result.min_deviation
    assert 0 <= result.min_angle_step < result.angles.size
    assert 0 <= result.max_angle_step < result.angles.size


def test_analyze_transmission_default_iterations_uses_rotation_period():
    lk = _make_fourbar()
    expected = lk.get_rotation_period()
    result = analyze_transmission(lk)
    assert result.angles.size == expected


def test_transmission_acceptable_range_property():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12)
    assert result.acceptable_range == (40.0, 140.0)


def test_transmission_worst_angle_far_from_90():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12)
    worst = result.worst_angle()
    # Worst is whichever of min/max has the larger |x - 90|
    dev_min = abs(result.min_angle - 90.0)
    dev_max = abs(result.max_angle - 90.0)
    if dev_max >= dev_min:
        assert worst == result.max_angle
    else:
        assert worst == result.min_angle


def test_analyze_transmission_is_acceptable_flag():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12, acceptable_range=(0.0, 180.0))
    assert result.is_acceptable is True

    strict = analyze_transmission(lk, iterations=12, acceptable_range=(89.0, 91.0))
    assert strict.is_acceptable is False


def test_analyze_transmission_restores_state_after_run():
    lk = _make_fourbar()
    initial = lk.get_coords()
    analyze_transmission(lk, iterations=8)
    restored = lk.get_coords()
    for (x0, y0), (x1, y1) in zip(initial, restored):
        if x0 is None or x1 is None:
            continue
        assert math.isclose(x0, x1, abs_tol=1e-9)
        assert math.isclose(y0, y1, abs_tol=1e-9)


def test_transmission_plot_creates_axes():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12)
    ax = result.plot()
    assert ax is not None
    # Check reasonable content
    assert ax.get_ylim() == (0.0, 180.0)
    assert ax.get_xlabel() == "Crank angle (degrees)"
    plt.close("all")


def test_transmission_plot_with_supplied_axes_and_options():
    lk = _make_fourbar()
    result = analyze_transmission(lk, iterations=12)
    fig, ax = plt.subplots()
    returned = result.plot(ax=ax, show_limits=False, show_optimum=False, title=None)
    assert returned is ax
    assert ax.get_title() == ""
    plt.close(fig)


# ---------------------------------------------------------------------------
# compute_slide_position
# ---------------------------------------------------------------------------


def test_compute_slide_position_on_axis():
    pos = compute_slide_position((3.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(pos, 3.0)


def test_compute_slide_position_off_axis_projects():
    # Slider at (2, 5), axis along +x through origin
    pos = compute_slide_position((2.0, 5.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(pos, 2.0)


def test_compute_slide_position_signed_negative():
    pos = compute_slide_position((-2.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isclose(pos, -2.0)


def test_compute_slide_position_degenerate_line():
    # Line points coincide - returns 0.0
    assert compute_slide_position((5.0, 5.0), (1.0, 1.0), (1.0, 1.0)) == 0.0


def test_compute_slide_position_diagonal_axis():
    # Axis is 45 degrees. Project (1, 0) onto it -> 1/sqrt(2)
    pos = compute_slide_position((1.0, 0.0), (0.0, 0.0), (1.0, 1.0))
    assert math.isclose(pos, 1.0 / math.sqrt(2), abs_tol=1e-9)


# ---------------------------------------------------------------------------
# _auto_detect_prismatic_joint
# ---------------------------------------------------------------------------


def test_auto_detect_prismatic_joint_finds_it():
    lk = _make_slider_crank()
    joint = _auto_detect_prismatic_joint(lk)
    assert type(joint).__name__ == "RRPDyad"


def test_auto_detect_prismatic_joint_missing_raises():
    lk = _make_fourbar()
    with pytest.raises(ValueError, match="no Prismatic"):
        _auto_detect_prismatic_joint(lk)


# ---------------------------------------------------------------------------
# stroke_at_position
# ---------------------------------------------------------------------------


class _MockPoint:
    """Legacy-style point with x/y attributes."""

    def __init__(self, x: float, y: float, name: str = "") -> None:
        self.x = x
        self.y = y
        self.name = name


class _MockPrismaticJoint:
    """Emulates the legacy Prismatic joint interface required by stroke
    analysis (joint1 / joint2 attributes)."""

    def __init__(self, x: float, y: float, p1: _MockPoint, p2: _MockPoint) -> None:
        self.x = x
        self.y = y
        self.joint1 = p1
        self.joint2 = p2
        self.name = "slider"

    @classmethod
    def __name__(cls):
        return "Prismatic"


# Make the class name "Prismatic" for auto-detection
_MockPrismaticJoint.__name__ = "Prismatic"


class _MockLinkage:
    """Minimal linkage substitute for stroke analysis tests."""

    def __init__(self, parts, frames) -> None:
        # parts list; auto-detector uses either `.joints` or `.components`
        self.components = parts
        self._frames = frames
        self._saved_coords = [(getattr(p, "x", 0.0), getattr(p, "y", 0.0)) for p in parts]

    def get_coords(self):
        return [(getattr(p, "x", None), getattr(p, "y", None)) for p in self.components]

    def set_coords(self, coords):
        for p, c in zip(self.components, coords):
            if hasattr(p, "x"):
                p.x = c[0]
                p.y = c[1]

    def get_rotation_period(self):
        return len(self._frames)

    def step(self, iterations=None, dt=1):
        n = iterations if iterations is not None else len(self._frames)
        for i in range(n):
            frame = self._frames[i % len(self._frames)]
            # Apply positions to parts so _get_joint_coord works
            for part, pos in zip(self.components, frame):
                if hasattr(part, "x"):
                    part.x = pos[0]
                    part.y = pos[1]
            yield tuple(frame)


def _make_mock_slider_linkage():
    """Linkage with a single oscillating slider along the x-axis."""
    p1 = _MockPoint(0.0, 0.0, name="P1")
    p2 = _MockPoint(1.0, 0.0, name="P2")
    slider = _MockPrismaticJoint(2.0, 0.0, p1, p2)
    # Frames: p1 and p2 stay; slider x moves from 1 to 5
    positions = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
    frames = [
        [(0.0, 0.0), (1.0, 0.0), (x, 0.0)] for x in positions
    ]
    return _MockLinkage([p1, p2, slider], frames), slider


def test_stroke_at_position_with_mock():
    lk, slider = _make_mock_slider_linkage()
    pos = stroke_at_position(lk)
    assert math.isclose(pos, 2.0)  # slider.x == 2.0 initially


def test_stroke_at_position_explicit_mock_joint():
    lk, slider = _make_mock_slider_linkage()
    pos = stroke_at_position(lk, prismatic_joint=slider)
    assert math.isclose(pos, 2.0)


def test_stroke_at_position_on_real_rrpdyad_raises_missing_joint1():
    """RRPDyad has line_anchor1/2 but not joint1/2, so the legacy-targeted
    stroke_at_position should raise a ValueError about missing joint1/joint2."""
    lk = _make_slider_crank()
    next(lk.step(iterations=1))
    slider = lk.components[-1]
    with pytest.raises(ValueError, match="joint1 and joint2"):
        stroke_at_position(lk, prismatic_joint=slider)


def test_stroke_at_position_auto_detect_failure():
    lk = _make_fourbar()
    # No prismatic joint -> auto-detect raises
    with pytest.raises(ValueError, match="no Prismatic"):
        stroke_at_position(lk)


def test_stroke_at_position_bad_slider_coord():
    class BadSlider:
        x = None
        y = None
        joint1 = _MockPoint(0.0, 0.0)
        joint2 = _MockPoint(1.0, 0.0)

    with pytest.raises(ValueError, match="slider position"):
        stroke_at_position(None, prismatic_joint=BadSlider())


def test_stroke_at_position_bad_line_coord():
    class BadLine:
        x = None
        y = None

    class Slider:
        x = 1.0
        y = 0.0
        joint1 = BadLine()
        joint2 = _MockPoint(1.0, 0.0)

    with pytest.raises(ValueError, match="line point positions"):
        stroke_at_position(None, prismatic_joint=Slider())


# ---------------------------------------------------------------------------
# analyze_stroke + StrokeAnalysis
# ---------------------------------------------------------------------------


def test_analyze_stroke_returns_dataclass():
    lk, _ = _make_mock_slider_linkage()
    result = analyze_stroke(lk, iterations=8)
    assert isinstance(result, StrokeAnalysis)
    assert result.min_position <= result.mean_position <= result.max_position
    assert math.isclose(
        result.stroke_range, result.max_position - result.min_position, abs_tol=1e-9
    )
    assert result.positions.size == 8
    assert 0 <= result.min_position_step < result.positions.size
    assert 0 <= result.max_position_step < result.positions.size


def test_analyze_stroke_default_iterations_uses_rotation_period():
    lk, _ = _make_mock_slider_linkage()
    expected = lk.get_rotation_period()
    result = analyze_stroke(lk)
    assert result.positions.size == expected


def test_analyze_stroke_values_match_input():
    lk, slider = _make_mock_slider_linkage()
    result = analyze_stroke(lk, iterations=8)
    # positions: 1, 2, 3, 4, 5, 4, 3, 2
    assert math.isclose(result.min_position, 1.0)
    assert math.isclose(result.max_position, 5.0)
    assert math.isclose(result.stroke_range, 4.0)


def test_analyze_stroke_explicit_joint():
    lk, slider = _make_mock_slider_linkage()
    result = analyze_stroke(lk, prismatic_joint=slider, iterations=4)
    assert result.positions.size == 4


def test_analyze_stroke_amplitude_and_center():
    lk, _ = _make_mock_slider_linkage()
    result = analyze_stroke(lk, iterations=8)
    assert math.isclose(result.amplitude, result.stroke_range / 2.0)
    assert math.isclose(
        result.center_position, (result.min_position + result.max_position) / 2.0
    )


def test_analyze_stroke_missing_joint1_raises():
    class BadSlider:
        x = 1.0
        y = 0.0
        # no joint1/joint2

    with pytest.raises(ValueError, match="joint1 and joint2"):
        analyze_stroke(None, prismatic_joint=BadSlider())
