"""Thorough tests for per-joint numba solvers (joints.py).

Targets all solver functions with analytically verifiable setups,
edge cases, NaN handling, and boundary conditions.
"""

import math

import pytest

from pylinkage.solver.joints import (
    solve_arc_crank,
    solve_crank,
    solve_fixed,
    solve_line_line,
    solve_linear,
    solve_linear_actuator,
    solve_oscillating_cam_follower,
    solve_revolute,
    solve_translating_cam_follower,
)


# ---------------------------------------------------------------------------
# solve_crank
# ---------------------------------------------------------------------------
class TestSolveCrank:
    """Tests for solve_crank (polar rotation around anchor)."""

    def test_quarter_rotation_from_origin(self):
        """Crank at (5,0) around origin, rotate 90 degrees -> (0,5)."""
        x, y = solve_crank(5.0, 0.0, 0.0, 0.0, 5.0, math.pi / 2, 1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(5.0, abs=1e-10)

    def test_half_rotation(self):
        """180-degree rotation flips to opposite side."""
        x, y = solve_crank(3.0, 0.0, 0.0, 0.0, 3.0, math.pi, 1.0)
        assert x == pytest.approx(-3.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_full_rotation_returns_to_start(self):
        """360-degree rotation returns to original position."""
        x, y = solve_crank(5.0, 0.0, 0.0, 0.0, 5.0, 2 * math.pi, 1.0)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_with_offset_anchor(self):
        """Anchor not at origin."""
        x, y = solve_crank(15.0, 10.0, 10.0, 10.0, 5.0, math.pi / 2, 1.0)
        assert x == pytest.approx(10.0, abs=1e-10)
        assert y == pytest.approx(15.0, abs=1e-10)

    def test_zero_dt_no_movement(self):
        """dt=0 means no rotation occurs."""
        x, y = solve_crank(5.0, 0.0, 0.0, 0.0, 5.0, 100.0, 0.0)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_negative_angle_rate(self):
        """Negative angle rate rotates clockwise."""
        x, y = solve_crank(5.0, 0.0, 0.0, 0.0, 5.0, -math.pi / 2, 1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-5.0, abs=1e-10)

    def test_small_angle_step(self):
        """Small angle step gives expected position."""
        dt = 0.01
        omega = 1.0
        x, y = solve_crank(5.0, 0.0, 0.0, 0.0, 5.0, omega, dt)
        expected_angle = omega * dt
        assert x == pytest.approx(5.0 * math.cos(expected_angle), abs=1e-10)
        assert y == pytest.approx(5.0 * math.sin(expected_angle), abs=1e-10)

    def test_starting_at_45_degrees(self):
        """Start at 45 degrees, rotate another 45 degrees -> 90 degrees."""
        r = 5.0
        start_x = r * math.cos(math.pi / 4)
        start_y = r * math.sin(math.pi / 4)
        x, y = solve_crank(start_x, start_y, 0.0, 0.0, r, math.pi / 4, 1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(r, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_revolute
# ---------------------------------------------------------------------------
class TestSolveRevolute:
    """Tests for solve_revolute (circle-circle intersection)."""

    def test_symmetric_intersection_upper(self):
        """Two equal circles at (0,0) and (6,0) with r=5 intersect near (3,4)."""
        x, y = solve_revolute(3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 5.0)
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(4.0, abs=1e-10)

    def test_symmetric_intersection_lower(self):
        """Same circles, hint toward lower intersection."""
        x, y = solve_revolute(3.0, -4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 5.0)
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(-4.0, abs=1e-10)

    def test_tangent_circles_single_intersection(self):
        """Externally tangent circles: centers at distance r0+r1."""
        x, y = solve_revolute(3.0, 0.0, 0.0, 0.0, 3.0, 5.0, 0.0, 2.0)
        assert x == pytest.approx(3.0, abs=1e-5)
        assert y == pytest.approx(0.0, abs=1e-5)

    def test_non_intersecting_circles_return_nan(self):
        """Circles too far apart return NaN."""
        x, y = solve_revolute(0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 1.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_one_circle_inside_other_return_nan(self):
        """Inner circle entirely inside outer circle."""
        x, y = solve_revolute(0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 0.0, 1.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_concentric_circles_return_nan(self):
        """Concentric circles (same center, different radii)."""
        x, y = solve_revolute(0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_asymmetric_radii(self):
        """Unequal radii: (0,0) r=3, (4,0) r=5 intersect at known points."""
        # d=4, intersection x = (d^2 + r0^2 - r1^2)/(2d) = (16+9-25)/8 = 0
        # y = sqrt(r0^2 - x^2) = 3
        x, y = solve_revolute(0.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_hysteresis_picks_nearest(self):
        """With two solutions, the one nearest to current position is picked."""
        # Circles at (0,0) r=5 and (6,0) r=5 have intersections at (3,4) and (3,-4)
        # Current pos near (3,4) -> pick upper
        x1, y1 = solve_revolute(3.0, 3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 5.0)
        assert y1 > 0
        # Current pos near (3,-4) -> pick lower
        x2, y2 = solve_revolute(3.0, -3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 5.0)
        assert y2 < 0


# ---------------------------------------------------------------------------
# solve_fixed
# ---------------------------------------------------------------------------
class TestSolveFixed:
    """Tests for solve_fixed (polar projection from parent line)."""

    def test_zero_angle_along_parent_direction(self):
        """Angle=0 projects along the p0->p1 direction."""
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 2.0, 0.0)
        assert x == pytest.approx(2.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_offset(self):
        """Angle=pi/2 projects perpendicular to p0->p1."""
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 1.0, math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.0, abs=1e-10)

    def test_180_degree_offset(self):
        """Angle=pi projects opposite to p0->p1."""
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 3.0, math.pi)
        assert x == pytest.approx(-3.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_diagonal_parent_direction(self):
        """Parent direction is 45 degrees, angle adds 45 more -> 90 deg total."""
        x, y = solve_fixed(0.0, 0.0, 1.0, 1.0, 2.0, math.pi / 4)
        # base_angle = pi/4, total = pi/2
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(2.0, abs=1e-10)

    def test_with_offset_origin(self):
        """Non-zero p0 position."""
        x, y = solve_fixed(5.0, 5.0, 6.0, 5.0, 3.0, 0.0)
        assert x == pytest.approx(8.0, abs=1e-10)
        assert y == pytest.approx(5.0, abs=1e-10)

    def test_negative_angle(self):
        """Negative angle rotates clockwise from parent direction."""
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 1.0, -math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-1.0, abs=1e-10)

    def test_parent_direction_vertical(self):
        """Parent direction is vertical (p1 directly above p0)."""
        x, y = solve_fixed(0.0, 0.0, 0.0, 1.0, 2.0, 0.0)
        # base_angle = pi/2
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_linear_actuator
# ---------------------------------------------------------------------------
class TestSolveLinearActuator:
    """Tests for solve_linear_actuator (oscillating linear motion)."""

    def test_basic_extension(self):
        """Simple extension along x-axis."""
        x, y, ext, d = solve_linear_actuator(0.5, 1.0, 0.0, 0.0, 0.0, 2.0, 0.1, 1.0)
        assert ext == pytest.approx(0.6, abs=1e-10)
        assert d == pytest.approx(1.0)
        assert x == pytest.approx(0.6, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_basic_retraction(self):
        """Retracting (direction=-1)."""
        x, y, ext, d = solve_linear_actuator(1.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.3, 1.0)
        assert ext == pytest.approx(0.7, abs=1e-10)
        assert d == pytest.approx(-1.0)

    def test_bounce_at_max_stroke(self):
        """Overshoot max stroke reverses direction."""
        x, y, ext, d = solve_linear_actuator(1.8, 1.0, 0.0, 0.0, 0.0, 2.0, 0.3, 1.0)
        # new_ext = 1.8 + 0.3 = 2.1 -> bounce: 2.0 - 0.1 = 1.9
        assert ext == pytest.approx(1.9, abs=1e-10)
        assert d == pytest.approx(-1.0)

    def test_bounce_at_zero(self):
        """Overshoot zero reverses direction."""
        x, y, ext, d = solve_linear_actuator(0.1, -1.0, 0.0, 0.0, 0.0, 2.0, 0.3, 1.0)
        # new_ext = 0.1 - 0.3 = -0.2 -> bounce: 0.2
        assert ext == pytest.approx(0.2, abs=1e-10)
        assert d == pytest.approx(1.0)

    def test_angled_actuator_45_degrees(self):
        """Actuator at 45 degrees."""
        x, y, ext, d = solve_linear_actuator(0.0, 1.0, 0.0, 0.0, math.pi / 4, 10.0, 1.0, 1.0)
        assert ext == pytest.approx(1.0, abs=1e-10)
        expected = 1.0 / math.sqrt(2)
        assert x == pytest.approx(expected, abs=1e-10)
        assert y == pytest.approx(expected, abs=1e-10)

    def test_angled_actuator_90_degrees(self):
        """Actuator at 90 degrees extends along y-axis."""
        x, y, ext, d = solve_linear_actuator(0.0, 1.0, 3.0, 4.0, math.pi / 2, 5.0, 1.0, 1.0)
        assert ext == pytest.approx(1.0, abs=1e-10)
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(5.0, abs=1e-10)

    def test_clamping_for_large_dt(self):
        """Very large dt that overshoots after bounce gets clamped."""
        x, y, ext, d = solve_linear_actuator(0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 500.0, 1.0)
        # After bounce, extension still overshoots -> clamped to [0, stroke]
        assert 0.0 <= ext <= 1.0

    def test_exactly_at_stroke(self):
        """Extension exactly reaches stroke."""
        x, y, ext, d = solve_linear_actuator(1.5, 1.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1.0)
        # new_ext = 1.5 + 0.5 = 2.0 exactly -> bounce: 2.0 - 0 = 2.0, dir = -1
        assert ext == pytest.approx(2.0, abs=1e-10)
        assert d == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# solve_arc_crank
# ---------------------------------------------------------------------------
class TestSolveArcCrank:
    """Tests for solve_arc_crank (oscillating angular motion with limits)."""

    def test_basic_rotation_within_bounds(self):
        """Normal rotation within arc limits."""
        x, y, angle, d = solve_arc_crank(
            0.5, 1.0, 0.0, 0.0, 1.0, 0.1, 0.0, math.pi, 1.0
        )
        assert angle == pytest.approx(0.6, abs=1e-10)
        assert d == pytest.approx(1.0)
        assert x == pytest.approx(math.cos(0.6), abs=1e-10)
        assert y == pytest.approx(math.sin(0.6), abs=1e-10)

    def test_bounce_at_arc_end(self):
        """Overshoot arc_end reverses direction."""
        x, y, angle, d = solve_arc_crank(
            2.9, 1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 3.0, 1.0
        )
        # new_angle = 2.9 + 0.2 = 3.1 -> 3.0 - 0.1 = 2.9
        assert angle == pytest.approx(2.9, abs=1e-10)
        assert d == pytest.approx(-1.0)

    def test_bounce_at_arc_start(self):
        """Overshoot arc_start reverses direction."""
        x, y, angle, d = solve_arc_crank(
            0.1, -1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 3.0, 1.0
        )
        # new_angle = 0.1 - 0.2 = -0.1 -> 0.0 + 0.1 = 0.1
        assert angle == pytest.approx(0.1, abs=1e-10)
        assert d == pytest.approx(1.0)

    def test_with_offset_anchor(self):
        """Arc crank around non-origin anchor."""
        x, y, angle, d = solve_arc_crank(
            0.0, 1.0, 10.0, 5.0, 3.0, math.pi / 6, -math.pi, math.pi, 1.0
        )
        expected_angle = math.pi / 6
        assert angle == pytest.approx(expected_angle, abs=1e-10)
        assert x == pytest.approx(10.0 + 3.0 * math.cos(expected_angle), abs=1e-10)
        assert y == pytest.approx(5.0 + 3.0 * math.sin(expected_angle), abs=1e-10)

    def test_clamping_large_dt(self):
        """Very large dt that overshoots after bounce gets clamped."""
        x, y, angle, d = solve_arc_crank(
            0.5, 1.0, 0.0, 0.0, 1.0, 1000.0, 0.0, 1.0, 1.0
        )
        assert 0.0 <= angle <= 1.0

    def test_exactly_at_arc_end(self):
        """Angle exactly reaches arc_end."""
        x, y, angle, d = solve_arc_crank(
            1.5, 1.0, 0.0, 0.0, 2.0, 0.5, 0.0, 2.0, 1.0
        )
        # new_angle = 1.5 + 0.5 = 2.0 exactly
        assert angle == pytest.approx(2.0, abs=1e-10)
        assert d == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# solve_linear
# ---------------------------------------------------------------------------
class TestSolveLinear:
    """Tests for solve_linear (circle-line intersection)."""

    def test_horizontal_line_two_intersections(self):
        """Circle at origin r=5, horizontal line y=3 -> x = +/- 4."""
        x, y = solve_linear(4.0, 3.0, 0.0, 0.0, 5.0, -10.0, 3.0, 10.0, 3.0)
        assert y == pytest.approx(3.0, abs=1e-10)
        assert x == pytest.approx(4.0, abs=1e-10)

    def test_horizontal_line_pick_negative(self):
        """Same setup but hint near negative x."""
        x, y = solve_linear(-4.0, 3.0, 0.0, 0.0, 5.0, -10.0, 3.0, 10.0, 3.0)
        assert y == pytest.approx(3.0, abs=1e-10)
        assert x == pytest.approx(-4.0, abs=1e-10)

    def test_tangent_line(self):
        """Line tangent to circle gives single intersection."""
        # Circle at origin r=1, line at y=1 -> tangent at (0,1)
        x, y = solve_linear(0.0, 1.0, 0.0, 0.0, 1.0, -5.0, 1.0, 5.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-5)
        assert y == pytest.approx(1.0, abs=1e-5)

    def test_no_intersection(self):
        """Line too far from circle center -> NaN."""
        x, y = solve_linear(0.0, 10.0, 0.0, 0.0, 1.0, -5.0, 10.0, 5.0, 10.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_vertical_line(self):
        """Circle at origin r=5, vertical line x=3."""
        # Intersection: x=3, y = +/- 4
        x, y = solve_linear(3.0, 4.0, 0.0, 0.0, 5.0, 3.0, -10.0, 3.0, 10.0)
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(4.0, abs=1e-10)

    def test_diagonal_line(self):
        """Circle at origin r=1, line y=x."""
        # Intersection at (1/sqrt(2), 1/sqrt(2)) or (-1/sqrt(2), -1/sqrt(2))
        v = 1.0 / math.sqrt(2)
        x, y = solve_linear(v, v, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0)
        assert x == pytest.approx(v, abs=1e-5)
        assert y == pytest.approx(v, abs=1e-5)


# ---------------------------------------------------------------------------
# solve_line_line
# ---------------------------------------------------------------------------
class TestSolveLineLine:
    """Tests for solve_line_line (line-line intersection)."""

    def test_perpendicular_at_origin(self):
        """x-axis and y-axis intersect at origin."""
        x, y = solve_line_line(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_y_equals_x_and_y_equals_minus_x(self):
        """y=x and y=-x intersect at origin."""
        x, y = solve_line_line(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_known_intersection_point(self):
        """y=2x+1 and y=-x+4 intersect at (1,3)."""
        # y=2x+1: points (0,1) and (1,3)
        # y=-x+4: points (0,4) and (4,0)
        x, y = solve_line_line(0.0, 1.0, 1.0, 3.0, 0.0, 4.0, 4.0, 0.0)
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_parallel_lines_return_nan(self):
        """Parallel lines (same slope) return NaN."""
        x, y = solve_line_line(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_horizontal_and_vertical(self):
        """Horizontal y=5 and vertical x=3 intersect at (3,5)."""
        x, y = solve_line_line(0.0, 5.0, 10.0, 5.0, 3.0, 0.0, 3.0, 10.0)
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(5.0, abs=1e-10)

    def test_nearly_parallel_lines(self):
        """Lines with very small angle between them still intersect."""
        # y=0 and y=0.001*x -> intersect at origin
        x, y = solve_line_line(-1.0, 0.0, 1.0, 0.0, -1000.0, -1.0, 1000.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-5)
        assert y == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# solve_translating_cam_follower
# ---------------------------------------------------------------------------
class TestSolveTranslatingCamFollower:
    """Tests for solve_translating_cam_follower."""

    def test_zero_displacement(self):
        """Zero displacement -> follower at guide position."""
        x, y = solve_translating_cam_follower(5.0, 3.0, 0.0, 0.0)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_horizontal_displacement(self):
        """Displacement along x-axis (angle=0)."""
        x, y = solve_translating_cam_follower(0.0, 0.0, 0.0, 7.0)
        assert x == pytest.approx(7.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_vertical_displacement(self):
        """Displacement along y-axis (angle=pi/2)."""
        x, y = solve_translating_cam_follower(1.0, 2.0, math.pi / 2, 4.0)
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(6.0, abs=1e-10)

    def test_negative_displacement(self):
        """Negative displacement moves backward from guide."""
        x, y = solve_translating_cam_follower(0.0, 0.0, 0.0, -3.0)
        assert x == pytest.approx(-3.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_diagonal_angle(self):
        """45-degree angle with displacement."""
        d = 2.0
        x, y = solve_translating_cam_follower(0.0, 0.0, math.pi / 4, d)
        expected = d / math.sqrt(2)
        assert x == pytest.approx(expected, abs=1e-10)
        assert y == pytest.approx(expected, abs=1e-10)

    def test_with_guide_offset(self):
        """Guide at non-origin position."""
        x, y = solve_translating_cam_follower(10.0, 20.0, math.pi, 5.0)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(20.0, abs=1e-10)


# ---------------------------------------------------------------------------
# solve_oscillating_cam_follower
# ---------------------------------------------------------------------------
class TestSolveOscillatingCamFollower:
    """Tests for solve_oscillating_cam_follower."""

    def test_zero_angle(self):
        """Arm at angle=0 points along positive x."""
        x, y = solve_oscillating_cam_follower(0.0, 0.0, 5.0, 0.0)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_90_degrees(self):
        """Arm at 90 degrees points along positive y."""
        x, y = solve_oscillating_cam_follower(0.0, 0.0, 3.0, math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_180_degrees(self):
        """Arm at 180 degrees points along negative x."""
        x, y = solve_oscillating_cam_follower(0.0, 0.0, 4.0, math.pi)
        assert x == pytest.approx(-4.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_with_offset_pivot(self):
        """Non-origin pivot."""
        x, y = solve_oscillating_cam_follower(10.0, 20.0, 1.0, 0.0)
        assert x == pytest.approx(11.0, abs=1e-10)
        assert y == pytest.approx(20.0, abs=1e-10)

    def test_270_degrees(self):
        """Arm at 270 degrees points along negative y."""
        x, y = solve_oscillating_cam_follower(0.0, 0.0, 2.0, 3 * math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-2.0, abs=1e-10)

    def test_45_degrees_with_pivot_offset(self):
        """Arm at 45 degrees from offset pivot."""
        r = 4.0
        x, y = solve_oscillating_cam_follower(5.0, 5.0, r, math.pi / 4)
        expected_offset = r / math.sqrt(2)
        assert x == pytest.approx(5.0 + expected_offset, abs=1e-10)
        assert y == pytest.approx(5.0 + expected_offset, abs=1e-10)
