"""Tests for the rigid-body velocity and acceleration solvers.

These propagate the motion of a body from two of its points, rather than
intersecting two distance constraints. They exist to cover the case the
constraint-intersection form cannot: a point collinear with its two
anchors, where the Jacobian is singular but the motion is not ambiguous.
"""

import math

import pytest

from pylinkage.solver.acceleration import (
    solve_revolute_acceleration,
    solve_rigid_body_acceleration,
)
from pylinkage.solver.velocity import (
    solve_revolute_velocity,
    solve_rigid_body_velocity,
)


def _spin(point, omega, alpha=0.0, centre=(0.0, 0.0)):
    """Velocity and acceleration of *point* on a body spinning about *centre*."""
    rx, ry = point[0] - centre[0], point[1] - centre[1]
    velocity = (-omega * ry, omega * rx)
    acceleration = (
        -alpha * ry - omega * omega * rx,
        alpha * rx - omega * omega * ry,
    )
    return velocity, acceleration


# ---------------------------------------------------------------------------
# solve_rigid_body_velocity
# ---------------------------------------------------------------------------
class TestSolveRigidBodyVelocity:
    """Tests for solve_rigid_body_velocity."""

    def test_pure_rotation_recovers_tangential_velocity(self):
        """A body spinning about the origin: v = omega x r everywhere."""
        omega = 2.5
        p0, p1, query = (1.0, 0.0), (3.0, 0.0), (5.0, 0.0)
        v0, _ = _spin(p0, omega)
        v1, _ = _spin(p1, omega)

        vx, vy = solve_rigid_body_velocity(
            query[0], query[1], p0[0], p0[1], v0[0], v0[1], p1[0], p1[1], v1[0], v1[1],
        )
        expected, _ = _spin(query, omega)
        assert vx == pytest.approx(expected[0], abs=1e-12)
        assert vy == pytest.approx(expected[1], abs=1e-12)

    def test_pure_translation(self):
        """Equal point velocities mean zero omega and uniform motion."""
        vx, vy = solve_rigid_body_velocity(
            7.0, -4.0, 0.0, 0.0, 1.5, -0.5, 2.0, 3.0, 1.5, -0.5,
        )
        assert vx == pytest.approx(1.5, abs=1e-12)
        assert vy == pytest.approx(-0.5, abs=1e-12)

    def test_query_at_reference_point_returns_its_velocity(self):
        vx, vy = solve_rigid_body_velocity(
            1.0, 2.0, 1.0, 2.0, 0.3, 0.7, 4.0, 6.0, -0.2, 1.1,
        )
        assert vx == pytest.approx(0.3, abs=1e-12)
        assert vy == pytest.approx(0.7, abs=1e-12)

    def test_agrees_with_revolute_solver_when_not_collinear(self):
        """Both formulations describe the same body, so where the
        constraint-intersection form is well-conditioned they must match."""
        omega = -1.25
        p0, p1, query = (1.0, 0.0), (0.0, 2.0), (2.0, 3.0)
        v0, _ = _spin(p0, omega)
        v1, _ = _spin(p1, omega)

        rigid = solve_rigid_body_velocity(
            query[0], query[1], p0[0], p0[1], v0[0], v0[1], p1[0], p1[1], v1[0], v1[1],
        )
        revolute = solve_revolute_velocity(
            query[0], query[1], p0[0], p0[1], v0[0], v0[1], p1[0], p1[1], v1[0], v1[1],
        )
        assert not math.isnan(revolute[0])
        assert rigid[0] == pytest.approx(revolute[0], abs=1e-9)
        assert rigid[1] == pytest.approx(revolute[1], abs=1e-9)

    def test_resolves_collinear_case_that_defeats_revolute_solver(self):
        """The reason this solver exists."""
        omega = 0.8
        p0, p1, query = (1.0, 1.0), (3.0, 3.0), (5.0, 5.0)  # all on y = x
        v0, _ = _spin(p0, omega)
        v1, _ = _spin(p1, omega)

        revolute = solve_revolute_velocity(
            query[0], query[1], p0[0], p0[1], v0[0], v0[1], p1[0], p1[1], v1[0], v1[1],
        )
        assert math.isnan(revolute[0]) and math.isnan(revolute[1])

        vx, vy = solve_rigid_body_velocity(
            query[0], query[1], p0[0], p0[1], v0[0], v0[1], p1[0], p1[1], v1[0], v1[1],
        )
        expected, _ = _spin(query, omega)
        assert vx == pytest.approx(expected[0], abs=1e-12)
        assert vy == pytest.approx(expected[1], abs=1e-12)

    def test_coincident_reference_points_are_singular(self):
        """Two identical points cannot fix the body's rotation."""
        vx, vy = solve_rigid_body_velocity(
            5.0, 5.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        )
        assert math.isnan(vx) and math.isnan(vy)

    def test_nan_position_propagates(self):
        vx, vy = solve_rigid_body_velocity(
            math.nan, 5.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0,
        )
        assert math.isnan(vx) and math.isnan(vy)


# ---------------------------------------------------------------------------
# solve_rigid_body_acceleration
# ---------------------------------------------------------------------------
class TestSolveRigidBodyAcceleration:
    """Tests for solve_rigid_body_acceleration."""

    def test_constant_spin_is_purely_centripetal(self):
        """With alpha = 0, a = -omega**2 * r, pointing at the centre."""
        omega = 3.0
        p0, p1, query = (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)
        v0, a0 = _spin(p0, omega)
        v1, a1 = _spin(p1, omega)

        ax, ay = solve_rigid_body_acceleration(
            query[0], query[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        assert ax == pytest.approx(-omega * omega * query[0], abs=1e-12)
        assert ay == pytest.approx(0.0, abs=1e-12)

    def test_angular_acceleration_adds_tangential_term(self):
        omega, alpha = 1.5, 0.75
        p0, p1, query = (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)
        v0, a0 = _spin(p0, omega, alpha)
        v1, a1 = _spin(p1, omega, alpha)

        ax, ay = solve_rigid_body_acceleration(
            query[0], query[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        expected_v, expected_a = _spin(query, omega, alpha)
        assert ax == pytest.approx(expected_a[0], abs=1e-12)
        assert ay == pytest.approx(expected_a[1], abs=1e-12)

    def test_agrees_with_revolute_solver_when_not_collinear(self):
        omega, alpha = -0.9, 0.4
        p0, p1, query = (1.0, 0.0), (0.0, 2.0), (2.0, 3.0)
        v0, a0 = _spin(p0, omega, alpha)
        v1, a1 = _spin(p1, omega, alpha)
        vq, _ = _spin(query, omega, alpha)

        rigid = solve_rigid_body_acceleration(
            query[0], query[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        revolute = solve_revolute_acceleration(
            query[0], query[1], vq[0], vq[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        assert not math.isnan(revolute[0])
        assert rigid[0] == pytest.approx(revolute[0], abs=1e-9)
        assert rigid[1] == pytest.approx(revolute[1], abs=1e-9)

    def test_resolves_collinear_case_that_defeats_revolute_solver(self):
        omega, alpha = 0.8, -0.3
        p0, p1, query = (1.0, 1.0), (3.0, 3.0), (5.0, 5.0)
        v0, a0 = _spin(p0, omega, alpha)
        v1, a1 = _spin(p1, omega, alpha)
        vq, expected = _spin(query, omega, alpha)

        revolute = solve_revolute_acceleration(
            query[0], query[1], vq[0], vq[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        assert math.isnan(revolute[0]) and math.isnan(revolute[1])

        ax, ay = solve_rigid_body_acceleration(
            query[0], query[1],
            p0[0], p0[1], v0[0], v0[1], a0[0], a0[1],
            p1[0], p1[1], v1[0], v1[1], a1[0], a1[1],
        )
        assert ax == pytest.approx(expected[0], abs=1e-12)
        assert ay == pytest.approx(expected[1], abs=1e-12)

    def test_coincident_reference_points_are_singular(self):
        ax, ay = solve_rigid_body_acceleration(
            5.0, 5.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(ax) and math.isnan(ay)

    def test_nan_position_propagates(self):
        ax, ay = solve_rigid_body_acceleration(
            math.nan, 5.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        )
        assert math.isnan(ax) and math.isnan(ay)
