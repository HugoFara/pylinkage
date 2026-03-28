"""Extended tests for per-joint numba solvers (joints.py).

Covers: solve_linear_actuator, solve_arc_crank, solve_line_line,
solve_translating_cam_follower, solve_oscillating_cam_follower,
and additional edge cases for solve_revolute, solve_linear, solve_fixed.
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


class TestSolveLinearActuator:
    """Tests for solve_linear_actuator."""

    def test_basic_extension(self):
        """Actuator extends along the x-axis."""
        x, y, ext, direction = solve_linear_actuator(
            current_extension=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=2.0,
            velocity=0.1,
            dt=1.0,
        )
        assert ext == pytest.approx(0.6, abs=1e-10)
        assert direction == pytest.approx(1.0)
        assert x == pytest.approx(0.6, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_bounce_at_max_stroke(self):
        """Direction reverses when reaching max stroke."""
        x, y, ext, direction = solve_linear_actuator(
            current_extension=1.9,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=2.0,
            velocity=0.2,
            dt=1.0,
        )
        # new_extension = 1.9 + 0.2 = 2.1, bounces to 2.0 - 0.1 = 1.9
        assert ext == pytest.approx(1.9, abs=1e-10)
        assert direction == pytest.approx(-1.0)

    def test_bounce_at_zero(self):
        """Direction reverses when reaching zero extension."""
        x, y, ext, direction = solve_linear_actuator(
            current_extension=0.1,
            direction=-1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=2.0,
            velocity=0.2,
            dt=1.0,
        )
        # new_extension = 0.1 - 0.2 = -0.1, bounces to 0.1
        assert ext == pytest.approx(0.1, abs=1e-10)
        assert direction == pytest.approx(1.0)

    def test_angled_actuator(self):
        """Actuator at 90 degrees extends along y-axis."""
        x, y, ext, direction = solve_linear_actuator(
            current_extension=0.0,
            direction=1.0,
            anchor_x=5.0,
            anchor_y=3.0,
            angle=math.pi / 2,
            stroke=10.0,
            velocity=1.0,
            dt=1.0,
        )
        assert ext == pytest.approx(1.0, abs=1e-10)
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(4.0, abs=1e-10)

    def test_clamp_large_dt(self):
        """Large dt that overshoots both bounds gets clamped."""
        # With a huge velocity*dt that overshoots stroke after bounce
        x, y, ext, direction = solve_linear_actuator(
            current_extension=1.0,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=2.0,
            velocity=100.0,  # massive velocity
            dt=1.0,
        )
        # After bounce, extension might still be > stroke, gets clamped
        assert 0.0 <= ext <= 2.0


class TestSolveArcCrank:
    """Tests for solve_arc_crank."""

    def test_basic_rotation(self):
        """Arc crank rotates within bounds."""
        x, y, angle, direction = solve_arc_crank(
            current_angle=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.1,
            arc_start=0.0,
            arc_end=math.pi,
            dt=1.0,
        )
        assert angle == pytest.approx(0.6, abs=1e-10)
        assert direction == pytest.approx(1.0)
        assert x == pytest.approx(math.cos(0.6), abs=1e-10)
        assert y == pytest.approx(math.sin(0.6), abs=1e-10)

    def test_bounce_at_arc_end(self):
        """Direction reverses at arc_end."""
        x, y, angle, direction = solve_arc_crank(
            current_angle=2.9,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.2,
            arc_start=0.0,
            arc_end=3.0,
            dt=1.0,
        )
        # new_angle = 2.9 + 0.2 = 3.1, bounces to 3.0 - 0.1 = 2.9
        assert angle == pytest.approx(2.9, abs=1e-10)
        assert direction == pytest.approx(-1.0)

    def test_bounce_at_arc_start(self):
        """Direction reverses at arc_start."""
        x, y, angle, direction = solve_arc_crank(
            current_angle=0.1,
            direction=-1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.2,
            arc_start=0.0,
            arc_end=3.0,
            dt=1.0,
        )
        # new_angle = 0.1 - 0.2 = -0.1, bounces to 0.0 + 0.1 = 0.1
        assert angle == pytest.approx(0.1, abs=1e-10)
        assert direction == pytest.approx(1.0)

    def test_with_offset_anchor(self):
        """Arc crank with non-zero anchor."""
        x, y, angle, direction = solve_arc_crank(
            current_angle=0.0,
            direction=1.0,
            anchor_x=10.0,
            anchor_y=5.0,
            radius=2.0,
            angle_rate=math.pi / 4,
            arc_start=-math.pi,
            arc_end=math.pi,
            dt=1.0,
        )
        expected_angle = math.pi / 4
        assert angle == pytest.approx(expected_angle, abs=1e-10)
        assert x == pytest.approx(10.0 + 2.0 * math.cos(expected_angle), abs=1e-10)
        assert y == pytest.approx(5.0 + 2.0 * math.sin(expected_angle), abs=1e-10)


class TestSolveLineLine:
    """Tests for solve_line_line."""

    def test_perpendicular_lines(self):
        """Two perpendicular lines intersect at the origin."""
        x, y = solve_line_line(
            line1_p1_x=-1.0,
            line1_p1_y=0.0,
            line1_p2_x=1.0,
            line1_p2_y=0.0,
            line2_p1_x=0.0,
            line2_p1_y=-1.0,
            line2_p2_x=0.0,
            line2_p2_y=1.0,
        )
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_angled_lines(self):
        """Two angled lines intersect at a known point."""
        # Line 1: y = x (through (0,0) and (1,1))
        # Line 2: y = -x + 2 (through (0,2) and (2,0))
        # Intersection at (1, 1)
        x, y = solve_line_line(
            line1_p1_x=0.0,
            line1_p1_y=0.0,
            line1_p2_x=1.0,
            line1_p2_y=1.0,
            line2_p1_x=0.0,
            line2_p1_y=2.0,
            line2_p2_x=2.0,
            line2_p2_y=0.0,
        )
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(1.0, abs=1e-10)

    def test_parallel_lines_return_nan(self):
        """Parallel lines have no intersection -> NaN."""
        x, y = solve_line_line(
            line1_p1_x=0.0,
            line1_p1_y=0.0,
            line1_p2_x=1.0,
            line1_p2_y=0.0,
            line2_p1_x=0.0,
            line2_p1_y=1.0,
            line2_p2_x=1.0,
            line2_p2_y=1.0,
        )
        assert math.isnan(x)
        assert math.isnan(y)

    def test_offset_intersection(self):
        """Lines intersect at a non-trivial point."""
        # Line 1: horizontal at y=3, through (0,3) and (10,3)
        # Line 2: vertical at x=5, through (5,0) and (5,10)
        x, y = solve_line_line(
            line1_p1_x=0.0,
            line1_p1_y=3.0,
            line1_p2_x=10.0,
            line1_p2_y=3.0,
            line2_p1_x=5.0,
            line2_p1_y=0.0,
            line2_p2_x=5.0,
            line2_p2_y=10.0,
        )
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)


class TestSolveTranslatingCamFollower:
    """Tests for solve_translating_cam_follower."""

    def test_zero_displacement(self):
        """Zero displacement returns guide position."""
        x, y = solve_translating_cam_follower(
            guide_x=5.0,
            guide_y=3.0,
            guide_angle=0.0,
            displacement=0.0,
        )
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_horizontal_displacement(self):
        """Positive displacement along x-axis."""
        x, y = solve_translating_cam_follower(
            guide_x=0.0,
            guide_y=0.0,
            guide_angle=0.0,
            displacement=3.0,
        )
        assert x == pytest.approx(3.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_vertical_displacement(self):
        """Displacement along y-axis (angle=pi/2)."""
        x, y = solve_translating_cam_follower(
            guide_x=1.0,
            guide_y=2.0,
            guide_angle=math.pi / 2,
            displacement=4.0,
        )
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(6.0, abs=1e-10)

    def test_negative_displacement(self):
        """Negative displacement moves backward."""
        x, y = solve_translating_cam_follower(
            guide_x=0.0,
            guide_y=0.0,
            guide_angle=0.0,
            displacement=-2.0,
        )
        assert x == pytest.approx(-2.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)


class TestSolveOscillatingCamFollower:
    """Tests for solve_oscillating_cam_follower."""

    def test_zero_angle(self):
        """Arm at angle 0 points along x-axis."""
        x, y = solve_oscillating_cam_follower(
            pivot_x=0.0,
            pivot_y=0.0,
            arm_length=5.0,
            arm_angle=0.0,
        )
        assert x == pytest.approx(5.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_angle(self):
        """Arm at 90 degrees points along y-axis."""
        x, y = solve_oscillating_cam_follower(
            pivot_x=0.0,
            pivot_y=0.0,
            arm_length=3.0,
            arm_angle=math.pi / 2,
        )
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(3.0, abs=1e-10)

    def test_with_offset_pivot(self):
        """Arm with offset pivot."""
        x, y = solve_oscillating_cam_follower(
            pivot_x=10.0,
            pivot_y=20.0,
            arm_length=1.0,
            arm_angle=math.pi,
        )
        assert x == pytest.approx(9.0, abs=1e-10)
        assert y == pytest.approx(20.0, abs=1e-10)


class TestSolveRevoluteEdgeCases:
    """Additional edge cases for solve_revolute."""

    def test_picks_nearest_to_current_position(self):
        """When two intersections exist, pick the one nearest to current."""
        # Two circles intersect at (1.5, ~1.32) and (1.5, ~-1.32)
        # Current position near lower -> should pick lower
        x, y = solve_revolute(
            current_x=1.5,
            current_y=-1.0,
            p0_x=0.0,
            p0_y=0.0,
            r0=2.0,
            p1_x=3.0,
            p1_y=0.0,
            r1=2.0,
        )
        assert x == pytest.approx(1.5, abs=1e-10)
        assert y < 0  # Lower intersection

    def test_concentric_circles_no_intersection(self):
        """Concentric circles with different radii have no intersection."""
        x, y = solve_revolute(
            current_x=0.0,
            current_y=0.0,
            p0_x=0.0,
            p0_y=0.0,
            r0=1.0,
            p1_x=0.0,
            p1_y=0.0,
            r1=2.0,
        )
        assert math.isnan(x)
        assert math.isnan(y)


class TestSolveCrankEdgeCases:
    """Additional tests for solve_crank."""

    def test_zero_dt(self):
        """Zero time step means no rotation."""
        x, y = solve_crank(
            current_x=1.0,
            current_y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=math.pi,
            dt=0.0,
        )
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_full_rotation(self):
        """Full 2*pi rotation returns to start."""
        x, y = solve_crank(
            current_x=1.0,
            current_y=0.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=2.0 * math.pi,
            dt=1.0,
        )
        assert x == pytest.approx(1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)


class TestSolveFixedEdgeCases:
    """Additional tests for solve_fixed."""

    def test_negative_angle(self):
        """Negative angle offset."""
        x, y = solve_fixed(
            p0_x=0.0,
            p0_y=0.0,
            p1_x=1.0,
            p1_y=0.0,
            radius=1.0,
            angle=-math.pi / 2,
        )
        # -90 degrees from the line pointing right should point down
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-1.0, abs=1e-10)

    def test_pi_angle(self):
        """180 degree angle offset points opposite direction."""
        x, y = solve_fixed(
            p0_x=0.0,
            p0_y=0.0,
            p1_x=1.0,
            p1_y=0.0,
            radius=1.0,
            angle=math.pi,
        )
        assert x == pytest.approx(-1.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)


class TestSolveLinearEdgeCases:
    """Additional tests for solve_linear."""

    def test_tangent_from_above(self):
        """Circle tangent to line, current position on tangent side."""
        x, y = solve_linear(
            current_x=0.0,
            current_y=2.0,
            circle_x=0.0,
            circle_y=0.0,
            radius=2.0,
            line_p1_x=-5.0,
            line_p1_y=2.0,
            line_p2_x=5.0,
            line_p2_y=2.0,
        )
        assert x == pytest.approx(0.0, abs=1e-5)
        assert y == pytest.approx(2.0, abs=1e-5)

    def test_diagonal_line(self):
        """Circle-line intersection with a diagonal line."""
        # Circle at origin radius 1, line y=x (through (0,0) and (1,1))
        x, y = solve_linear(
            current_x=0.5,
            current_y=0.5,
            circle_x=0.0,
            circle_y=0.0,
            radius=1.0,
            line_p1_x=0.0,
            line_p1_y=0.0,
            line_p2_x=1.0,
            line_p2_y=1.0,
        )
        # On line y=x and distance 1 from origin: x=y=1/sqrt(2) or x=y=-1/sqrt(2)
        # Current hint is positive, so pick positive
        expected = 1.0 / math.sqrt(2)
        assert x == pytest.approx(expected, abs=1e-5)
        assert y == pytest.approx(expected, abs=1e-5)
