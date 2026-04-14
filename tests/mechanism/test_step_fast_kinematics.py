"""Tests for step_fast_with_kinematics() on the modern containers.

Numba-batched simulation that also returns velocity and acceleration
trajectories. Tests parity with the per-step Python path in
step_with_derivatives and the analytic closed forms for a crank tip.
"""

import math

import numpy as np
import pytest

from pylinkage.mechanism import DriverLink, fourbar


def _modern_fourbar():
    from pylinkage.actuators import Crank
    from pylinkage.components import Ground
    from pylinkage.dyads import RRRDyad
    from pylinkage.simulation import Linkage

    A = Ground(0.0, 0.0, name="A")
    D = Ground(4.0, 0.0, name="D")
    crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="crank")
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=D,
        distance1=3.0,
        distance2=3.0,
        name="C",
    )
    return Linkage([A, D, crank, pin], name="modern-fourbar"), crank


class TestMechanismStepFastKinematics:
    def test_returns_three_trajectory_arrays(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        driver = next(link for link in m.links if isinstance(link, DriverLink))
        m.set_input_velocity(driver, omega=10.0)

        pos, vel, acc = m.step_fast_with_kinematics(iterations=10)
        shape = (10, len(m.joints), 2)
        assert pos.shape == shape
        assert vel.shape == shape
        assert acc.shape == shape
        for arr in (pos, vel, acc):
            assert arr.dtype == np.float64

    def test_crank_tip_speed_matches_omega_radius(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        driver = next(link for link in m.links if isinstance(link, DriverLink))
        omega = 10.0
        m.set_input_velocity(driver, omega=omega)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        _pos, vel, _acc = m.step_fast_with_kinematics(iterations=1)
        speed = math.hypot(vel[0, tip_idx, 0], vel[0, tip_idx, 1])
        assert speed == pytest.approx(omega * driver.radius, rel=1e-6)

    def test_crank_tip_centripetal_when_alpha_zero(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        driver = next(link for link in m.links if isinstance(link, DriverLink))
        omega = 5.0
        m.set_input_velocity(driver, omega=omega, alpha=0.0)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        _pos, _vel, acc = m.step_fast_with_kinematics(iterations=1)
        mag = math.hypot(acc[0, tip_idx, 0], acc[0, tip_idx, 1])
        assert mag == pytest.approx(omega * omega * driver.radius, rel=1e-5)


class TestSimulationLinkageStepFastKinematics:
    def test_returns_three_trajectory_arrays(self) -> None:
        linkage, crank = _modern_fourbar()
        linkage.set_input_velocity(crank, omega=10.0)
        pos, vel, acc = linkage.step_fast_with_kinematics(iterations=10)
        shape = (10, len(linkage.components), 2)
        assert pos.shape == shape
        assert vel.shape == shape
        assert acc.shape == shape

    def test_crank_tip_speed_matches_omega_radius(self) -> None:
        linkage, crank = _modern_fourbar()
        omega = 7.5
        linkage.set_input_velocity(crank, omega=omega)
        crank_idx = linkage.components.index(crank)
        _pos, vel, _acc = linkage.step_fast_with_kinematics(iterations=1)
        speed = math.hypot(vel[0, crank_idx, 0], vel[0, crank_idx, 1])
        assert speed == pytest.approx(omega * crank.radius, rel=1e-6)

    def test_default_iterations_uses_rotation_period(self) -> None:
        linkage, crank = _modern_fourbar()
        linkage.set_input_velocity(crank, omega=1.0)
        pos, _vel, _acc = linkage.step_fast_with_kinematics()
        assert pos.shape[0] == linkage.get_rotation_period()

    def test_zero_omega_gives_zero_tip_speed(self) -> None:
        linkage, crank = _modern_fourbar()
        # No set_input_velocity → omega defaults to 0.
        crank_idx = linkage.components.index(crank)
        _pos, vel, _acc = linkage.step_fast_with_kinematics(iterations=1)
        assert vel[0, crank_idx, 0] == pytest.approx(0.0)
        assert vel[0, crank_idx, 1] == pytest.approx(0.0)
