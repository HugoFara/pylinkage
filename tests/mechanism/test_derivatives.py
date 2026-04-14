"""Tests for Mechanism velocity/acceleration kinematics.

Covers ``set_input_velocity`` / ``get_velocities`` / ``get_accelerations`` /
``step_with_derivatives`` on the ``pylinkage.mechanism.Mechanism`` class.
"""

import math

import pytest

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    Mechanism,
    fourbar,
)


def _crank_and_mechanism() -> tuple[DriverLink, Mechanism]:
    """Build a Grashof four-bar and return its driver link plus mechanism."""
    m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
    driver = next(link for link in m.links if isinstance(link, DriverLink))
    return driver, m


# ---------------------------------------------------------------------------
# set_input_velocity
# ---------------------------------------------------------------------------


class TestSetInputVelocity:
    def test_stores_omega_on_driver(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        assert driver._omega == 10.0
        assert driver._alpha == 0.0

    def test_stores_alpha(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0, alpha=2.5)
        assert driver._alpha == 2.5

    def test_unknown_driver_raises(self) -> None:
        _, m = _crank_and_mechanism()
        # A driver link not registered with this mechanism.
        ground = GroundJoint(id="X", position=(0.0, 0.0))
        tip = GroundJoint(id="Y", position=(1.0, 0.0))
        rogue = DriverLink(id="rogue", joints=[ground, tip], motor_joint=ground)
        with pytest.raises(ValueError):
            m.set_input_velocity(rogue, omega=1.0)


# ---------------------------------------------------------------------------
# get_velocities / get_accelerations defaults
# ---------------------------------------------------------------------------


class TestKinematicAccessorDefaults:
    def test_velocities_none_before_step(self) -> None:
        _, m = _crank_and_mechanism()
        assert all(v is None for v in m.get_velocities())

    def test_accelerations_none_before_step(self) -> None:
        _, m = _crank_and_mechanism()
        assert all(a is None for a in m.get_accelerations())


# ---------------------------------------------------------------------------
# step_with_derivatives
# ---------------------------------------------------------------------------


class TestStepWithDerivatives:
    def test_yields_three_tuples_of_correct_length(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        n = len(m.joints)
        for pos, vel, acc in m.step_with_derivatives(iterations=3):
            assert len(pos) == n
            assert len(vel) == n
            assert len(acc) == n

    def test_grounds_have_zero_kinematics(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        ground_indices = [i for i, j in enumerate(m.joints) if isinstance(j, GroundJoint)]
        assert ground_indices, "Expected at least one ground joint"
        for _pos, vel, acc in m.step_with_derivatives(iterations=1):
            for i in ground_indices:
                assert vel[i] == (0.0, 0.0)
                assert acc[i] == (0.0, 0.0)

    def test_crank_tip_speed_matches_omega_radius(self) -> None:
        driver, m = _crank_and_mechanism()
        omega = 10.0
        m.set_input_velocity(driver, omega=omega)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            assert vel[tip_idx] is not None
            speed = math.hypot(*vel[tip_idx])
            assert speed == pytest.approx(omega * driver.radius, rel=1e-9)

    def test_dependent_joint_velocity_computed(self) -> None:
        """The coupler/rocker connection should pick up a finite velocity."""
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)

        non_ground_non_tip = [
            i
            for i, j in enumerate(m.joints)
            if not isinstance(j, GroundJoint) and j is not driver.output_joint
        ]
        assert non_ground_non_tip, "Need at least one driven joint"

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            for i in non_ground_non_tip:
                assert vel[i] is not None

    def test_acceleration_matches_centripetal_when_alpha_zero(self) -> None:
        driver, m = _crank_and_mechanism()
        omega = 5.0
        m.set_input_velocity(driver, omega=omega, alpha=0.0)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)
        ground = driver.motor_joint
        assert ground is not None

        for _pos, _vel, acc in m.step_with_derivatives(iterations=1):
            assert acc[tip_idx] is not None
            ax, ay = acc[tip_idx]
            mag = math.hypot(ax, ay)
            # |a| = omega² * r when alpha = 0 (pure centripetal)
            assert mag == pytest.approx(omega * omega * driver.radius, rel=1e-6)

    def test_default_omega_zero_yields_zero_tip_velocity(self) -> None:
        """No set_input_velocity → tip velocity should still resolve to zero."""
        driver, m = _crank_and_mechanism()
        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            assert vel[tip_idx] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Joint runtime fields are independent of equality
# ---------------------------------------------------------------------------


class TestJointKinematicFieldsDoNotBreakEquality:
    def test_equality_ignores_velocity(self) -> None:
        j1 = GroundJoint(id="A", position=(0.0, 0.0))
        j2 = GroundJoint(id="A", position=(1.0, 1.0))
        j1.velocity = (3.0, 4.0)
        assert j1 == j2  # equality is by id only

    def test_kinematics_default_to_none(self) -> None:
        j = GroundJoint(id="A", position=(0.0, 0.0))
        assert j.velocity is None
        assert j.acceleration is None
