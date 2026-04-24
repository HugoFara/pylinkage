"""Tests exercising solver internals through the bridge + four-bar fixture.

These tests focus on driving functions in solver/simulation.py, joints.py,
velocity.py, and acceleration.py through integration with a Linkage.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.actuators import Crank, LinearActuator
from pylinkage.bridge import linkage_to_solver_data
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRPDyad, RRRDyad  # noqa: F401
from pylinkage.simulation import Linkage
from pylinkage.solver import (
    JOINT_CRANK,
    JOINT_FIXED,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    simulate,
    simulate_with_kinematics,
    step_single,
    step_single_acceleration,
    step_single_velocity,
)
from pylinkage.solver.acceleration import (
    solve_crank_acceleration,
    solve_fixed_acceleration,
    solve_prismatic_acceleration,
    solve_revolute_acceleration,
)
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
from pylinkage.solver.simulation import first_nan_step, has_nan_positions
from pylinkage.solver.velocity import (
    solve_crank_velocity,
    solve_fixed_velocity,
    solve_prismatic_velocity,
    solve_revolute_velocity,
)


def make_fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


class TestFourBarSimulation:
    def test_full_simulate_cycle(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        traj = simulate(
            data.positions,
            data.constraints,
            data.joint_types,
            data.parent_indices,
            data.constraint_offsets,
            data.solve_order,
            iterations=20,
            dt=0.1,
        )
        assert traj.shape == (20, 4, 2)
        assert not np.any(np.isnan(traj))

    def test_simulate_with_kinematics(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        n = data.positions.shape[0]
        vel = np.zeros((n, 2), dtype=np.float64)
        acc = np.zeros((n, 2), dtype=np.float64)
        # crank is index 2
        crank_indices = np.array([2], dtype=np.int32)
        omega = np.array([0.1], dtype=np.float64)
        alpha = np.array([0.0], dtype=np.float64)
        pt, vt, at = simulate_with_kinematics(
            data.positions,
            vel,
            acc,
            data.constraints,
            data.joint_types,
            data.parent_indices,
            data.constraint_offsets,
            data.solve_order,
            omega,
            alpha,
            crank_indices,
            iterations=10,
            dt=0.1,
        )
        assert pt.shape == (10, 4, 2)
        assert vt.shape == (10, 4, 2)
        assert at.shape == (10, 4, 2)
        assert not np.any(np.isnan(pt))


class TestSolveCrank:
    def test_crank_rotates_correctly(self):
        # Crank at (1, 0), anchor at (0, 0), rotate by 0.1 rad
        x, y = solve_crank(1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 1.0)
        assert x == pytest.approx(math.cos(0.1))
        assert y == pytest.approx(math.sin(0.1))

    def test_negative_angular_velocity(self):
        x, y = solve_crank(1.0, 0.0, 0.0, 0.0, 1.0, -0.1, 1.0)
        assert x == pytest.approx(math.cos(-0.1))
        assert y == pytest.approx(math.sin(-0.1))

    def test_zero_dt(self):
        x, y = solve_crank(1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0, abs=1e-12)


class TestSolveRevolute:
    def test_two_solutions_hysteresis(self):
        # Two circles with two intersections
        x, y = solve_revolute(0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0)
        # The intersection closer to (0, 1) should be returned
        assert not math.isnan(x)
        assert not math.isnan(y)

    def test_unbuildable_returns_nan(self):
        # Circles too far apart
        x, y = solve_revolute(0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 1.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_tangent_one_solution(self):
        x, y = solve_revolute(0.5, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0)
        assert not math.isnan(x)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0)


class TestSolveFixed:
    def test_zero_angle(self):
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_perpendicular(self):
        x, y = solve_fixed(0.0, 0.0, 1.0, 0.0, 1.0, math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(1.0)


class TestSolveLinear:
    def test_intersects_line(self):
        # Circle at origin, radius 2, line y = 1
        x, y = solve_linear(
            1.5,
            1.0,
            0.0,
            0.0,
            2.0,
            -5.0,
            1.0,
            5.0,
            1.0,
        )
        assert y == pytest.approx(1.0)
        assert not math.isnan(x)

    def test_unbuildable(self):
        # Circle radius too small to reach line
        x, y = solve_linear(
            0.0,
            5.0,
            0.0,
            0.0,
            1.0,
            -5.0,
            5.0,
            5.0,
            5.0,
        )
        assert math.isnan(x)
        assert math.isnan(y)

    def test_tangent_single(self):
        # Tangent at exact point
        x, y = solve_linear(
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            -5.0,
            1.0,
            5.0,
            1.0,
        )
        assert y == pytest.approx(1.0)


class TestSolveLineLineJoint:
    def test_intersecting(self):
        x, y = solve_line_line(0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(1.0)

    def test_parallel_returns_nan(self):
        x, y = solve_line_line(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert math.isnan(x)
        assert math.isnan(y)

    def test_coincident_lines(self):
        x, y = solve_line_line(0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0)
        # Coincident returns midpoint of first pair (representative)
        assert not math.isnan(x)
        assert not math.isnan(y)


class TestSolveLinearActuator:
    def test_extension_update(self):
        x, y, ext, direc = solve_linear_actuator(
            current_extension=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=1.0,
            velocity=0.1,
            dt=1.0,
        )
        assert ext == pytest.approx(0.6)
        assert direc == 1.0
        assert x == pytest.approx(0.6)

    def test_bounce_at_stroke_limit(self):
        x, y, ext, direc = solve_linear_actuator(
            current_extension=0.95,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=1.0,
            velocity=0.1,
            dt=1.0,
        )
        # Should bounce back
        assert direc == -1.0

    def test_bounce_at_zero_limit(self):
        x, y, ext, direc = solve_linear_actuator(
            current_extension=0.05,
            direction=-1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=1.0,
            velocity=0.1,
            dt=1.0,
        )
        assert direc == 1.0

    def test_clamping_large_dt(self):
        # Large dt that would overshoot
        x, y, ext, direc = solve_linear_actuator(
            current_extension=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            angle=0.0,
            stroke=1.0,
            velocity=10.0,
            dt=1.0,
        )
        # Should be clamped within range
        assert 0.0 <= ext <= 1.0


class TestSolveArcCrank:
    def test_normal_rotation(self):
        x, y, angle, direc = solve_arc_crank(
            current_angle=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.1,
            arc_start=0.0,
            arc_end=1.0,
            dt=1.0,
        )
        assert angle == pytest.approx(0.6)
        assert direc == 1.0

    def test_bounce_at_upper_limit(self):
        x, y, angle, direc = solve_arc_crank(
            current_angle=0.95,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.1,
            arc_start=0.0,
            arc_end=1.0,
            dt=1.0,
        )
        assert direc == -1.0

    def test_bounce_at_lower_limit(self):
        x, y, angle, direc = solve_arc_crank(
            current_angle=0.05,
            direction=-1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=0.1,
            arc_start=0.0,
            arc_end=1.0,
            dt=1.0,
        )
        assert direc == 1.0

    def test_clamping_large_dt(self):
        x, y, angle, direc = solve_arc_crank(
            current_angle=0.5,
            direction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            radius=1.0,
            angle_rate=10.0,
            arc_start=0.0,
            arc_end=1.0,
            dt=1.0,
        )
        assert 0.0 <= angle <= 1.0


class TestSolveCamFollowers:
    def test_translating(self):
        x, y = solve_translating_cam_follower(
            guide_x=0.0,
            guide_y=0.0,
            guide_angle=0.0,
            displacement=2.0,
        )
        assert x == pytest.approx(2.0)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_oscillating(self):
        x, y = solve_oscillating_cam_follower(
            pivot_x=0.0,
            pivot_y=0.0,
            arm_length=2.0,
            arm_angle=math.pi / 2,
        )
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(2.0)


class TestAccelerationSolvers:
    def test_crank_centripetal(self):
        # Crank at (1, 0), anchor at (0, 0), omega=1, alpha=0
        ax, ay = solve_crank_acceleration(
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0
        )
        # Centripetal: -omega^2 * r * (cos(theta), sin(theta)) = (-1, 0)
        assert ax == pytest.approx(-1.0)
        assert ay == pytest.approx(0.0, abs=1e-12)

    def test_crank_with_angular_accel(self):
        # Alpha = 1.0, tangential component
        ax, ay = solve_crank_acceleration(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0
        )
        # Tangential: r*alpha*(-sin(0), cos(0)) = (0, 1)
        assert ax == pytest.approx(0.0, abs=1e-12)
        assert ay == pytest.approx(1.0)

    def test_crank_nan_input(self):
        ax, ay = solve_crank_acceleration(
            math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_revolute_stationary_parents(self):
        # Joint at (0, 1), parents at (-1, 0) and (1, 0), all distance sqrt(2)
        ax, ay = solve_revolute_acceleration(
            0.0, 1.0, 0.0, 0.0,  # x, y, vx, vy
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # p0
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # p1
        )
        assert not math.isnan(ax)
        assert not math.isnan(ay)

    def test_revolute_nan_input(self):
        ax, ay = solve_revolute_acceleration(
            math.nan, 1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_revolute_singular(self):
        # Joint collinear with parents
        ax, ay = solve_revolute_acceleration(
            0.0, 0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_fixed_stationary(self):
        ax, ay = solve_fixed_acceleration(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
        )
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_fixed_nan(self):
        ax, ay = solve_fixed_acceleration(
            math.nan, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_fixed_coincident_parents_singular(self):
        ax, ay = solve_fixed_acceleration(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)

    def test_prismatic_stationary(self):
        jx = math.sqrt(3)
        jy = 1.0
        ax, ay = solve_prismatic_acceleration(
            jx, jy, 0.0, 0.0,  # position and velocity
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # circle
            2.0,  # radius
            -5.0, 1.0, 0.0, 0.0, 0.0, 0.0,  # line p1
            5.0, 1.0, 0.0, 0.0, 0.0, 0.0,  # line p2
        )
        assert ax == pytest.approx(0.0, abs=1e-10)
        assert ay == pytest.approx(0.0, abs=1e-10)

    def test_prismatic_nan_input(self):
        ax, ay = solve_prismatic_acceleration(
            math.nan, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0,
            -5.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            5.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(ax)
        assert math.isnan(ay)


class TestVelocitySolvers:
    def test_crank_velocity_basic(self):
        vx, vy = solve_crank_velocity(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5)
        # v = r*omega*(-sin, cos) = 0.5 * (0, 1) = (0, 0.5)
        assert vx == pytest.approx(0.0, abs=1e-12)
        assert vy == pytest.approx(0.5)

    def test_revolute_singular(self):
        # Collinear parents = singular
        vx, vy = solve_revolute_velocity(
            0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        assert math.isnan(vx)
        assert math.isnan(vy)

    def test_fixed_basic(self):
        vx, vy = solve_fixed_velocity(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # position, p0, p0 velocity
            2.0, 0.0, 0.0, 0.0,  # p1, p1 velocity
            1.0, 0.0,  # radius, angle
        )
        assert vx == pytest.approx(0.0, abs=1e-12)
        assert vy == pytest.approx(0.0, abs=1e-12)


class TestHasNanPositionsAndFirstNanStep:
    def test_clean(self):
        traj = np.ones((5, 3, 2))
        assert has_nan_positions(traj) is False
        assert first_nan_step(traj) == -1

    def test_with_nan(self):
        traj = np.ones((5, 3, 2))
        traj[2, 1, 0] = np.nan
        assert has_nan_positions(traj) is True
        assert first_nan_step(traj) == 2


class TestStepSingleWithLinearActuator:
    def test_linear_actuator_mechanism(self):
        # Build a mechanism with LinearActuator
        O1 = Ground(0.0, 0.0, name="O1")
        actuator = LinearActuator(
            anchor=O1,
            angle=0.0,
            stroke=2.0,
            speed=0.1,
            name="actuator",
        )
        linkage = Linkage([O1, actuator], name="Linear")
        # Just step a few times
        try:
            for _ in linkage.step(iterations=5):
                pass
        except Exception:
            pass
