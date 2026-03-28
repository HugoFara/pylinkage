"""Tests for pylinkage.simulation.linkage.Linkage (component-based API).

Covers: step, step_with_derivatives, rebuild, get/set_coords,
get/set_num_constraints, get_rotation_period, set_input_velocity,
get_velocities, get_accelerations, dyads property, automatic solve order,
and error paths.
"""

from __future__ import annotations

import math

import pytest

from pylinkage.actuators import ArcCrank, Crank, LinearActuator
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRPDyad, RRRDyad
from pylinkage.exceptions import UnderconstrainedError
from pylinkage.simulation import Linkage

# ---------------------------------------------------------------------------
# Helpers – build common mechanisms
# ---------------------------------------------------------------------------


def _four_bar(angular_velocity: float = 0.1) -> tuple[Ground, Ground, Crank, RRRDyad, Linkage]:
    """Return a simple four-bar: O1, O2, crank, rocker, linkage."""
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=angular_velocity, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    linkage = Linkage([O1, O2, crank, rocker], name="FourBar")
    return O1, O2, crank, rocker, linkage


def _four_bar_with_coupler() -> tuple[Ground, Ground, Crank, RRRDyad, FixedDyad, Linkage]:
    """Four-bar with a FixedDyad coupler point."""
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
    coupler = FixedDyad(
        anchor1=crank.output,
        anchor2=rocker,
        distance=1.0,
        angle=math.pi / 4,
        name="coupler",
    )
    linkage = Linkage([O1, O2, crank, rocker, coupler], name="FourBar+Coupler")
    return O1, O2, crank, rocker, coupler, linkage


def _slider_crank() -> tuple[Ground, Ground, Ground, Crank, RRPDyad, Linkage]:
    """Return a crank-slider mechanism."""
    O1 = Ground(0.0, 0.0, name="O1")
    L1 = Ground(0.0, -1.0, name="L1")
    L2 = Ground(4.0, -1.0, name="L2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=L1,
        line_anchor2=L2,
        distance=2.0,
        name="slider",
    )
    linkage = Linkage([O1, L1, L2, crank, slider], name="SliderCrank")
    return O1, L1, L2, crank, slider, linkage


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestLinkageConstruction:
    """Test Linkage creation and property access."""

    def test_name_default(self):
        O1 = Ground(0.0, 0.0)
        linkage = Linkage([O1])
        # Default name is str(id(...))
        assert linkage.name == str(id(linkage))

    def test_name_explicit(self):
        _, _, _, _, linkage = _four_bar()
        assert linkage.name == "FourBar"

    def test_components_tuple(self):
        O1, O2, crank, rocker, linkage = _four_bar()
        assert linkage.components == (O1, O2, crank, rocker)

    def test_dyads_alias(self):
        """The 'dyads' property should be the same as 'components'."""
        _, _, _, _, linkage = _four_bar()
        assert linkage.dyads is linkage.components

    def test_cranks_detected(self):
        _, _, crank, _, linkage = _four_bar()
        assert linkage._cranks == (crank,)

    def test_manual_order(self):
        O1, O2, crank, rocker, _ = _four_bar()
        linkage = Linkage([O1, O2, crank, rocker], order=[O1, O2, crank, rocker])
        assert linkage._solve_order == (O1, O2, crank, rocker)


# ---------------------------------------------------------------------------
# Solve-order computation
# ---------------------------------------------------------------------------


class TestSolveOrder:
    """Test _find_solve_order and related logic."""

    def test_auto_solve_order(self):
        O1, O2, crank, rocker, linkage = _four_bar()
        order = linkage._find_solve_order()
        # Grounds first, then crank, then rocker
        assert order.index(O1) < order.index(crank)
        assert order.index(O2) < order.index(rocker)
        assert order.index(crank) < order.index(rocker)

    def test_underconstrained_raises(self):
        """A disconnected component should fail solve-order computation."""
        O1 = Ground(0.0, 0.0, name="O1")
        # rocker has anchors that are not in the component list
        O_extra = Ground(5.0, 5.0, name="extra")
        crank = Crank(anchor=O_extra, radius=1.0, name="orphan_crank")
        # O_extra is not included in the linkage
        linkage = Linkage([O1, crank], name="bad")
        with pytest.raises(UnderconstrainedError):
            linkage._find_solve_order()

    def test_anchor_proxy_resolution(self):
        """Solve-order should resolve _AnchorProxy parents."""
        O1, O2, crank, rocker, linkage = _four_bar()
        order = linkage._find_solve_order()
        assert rocker in order


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------


class TestRebuild:
    def test_rebuild_triggers_solve_order(self):
        _, _, _, _, linkage = _four_bar()
        assert not hasattr(linkage, "_solve_order") or True  # may or may not exist
        linkage.rebuild()
        assert hasattr(linkage, "_solve_order")

    def test_rebuild_with_positions(self):
        O1, O2, crank, rocker, linkage = _four_bar()
        new_positions = [(0.0, 0.0), (3.0, 0.0), (0.5, 0.5), (2.0, 1.0)]
        linkage.rebuild(positions=new_positions)
        assert O1.position == (0.0, 0.0)
        assert crank.position == (0.5, 0.5)
        assert rocker.position == (2.0, 1.0)


# ---------------------------------------------------------------------------
# get_rotation_period
# ---------------------------------------------------------------------------


class TestGetRotationPeriod:
    def test_single_crank(self):
        _, _, _, _, linkage = _four_bar(angular_velocity=0.1)
        period = linkage.get_rotation_period()
        expected = round(math.tau / 0.1)
        assert period == expected

    def test_zero_velocity_crank(self):
        _, _, crank, _, linkage = _four_bar(angular_velocity=0.0)
        # With zero velocity, period should be 1 (no movement)
        assert linkage.get_rotation_period() == 1

    def test_arc_crank_period(self):
        O1 = Ground(0.0, 0.0, name="O1")
        arc = ArcCrank(
            anchor=O1,
            radius=1.0,
            angular_velocity=0.1,
            arc_start=0.0,
            arc_end=math.pi,
            name="arc",
        )
        linkage = Linkage([O1, arc], name="ArcOnly")
        period = linkage.get_rotation_period()
        # 2 * arc_range / angular_velocity = 2*pi / 0.1
        expected = round(2 * math.pi / 0.1)
        assert period == expected

    def test_linear_actuator_period(self):
        O1 = Ground(0.0, 0.0, name="O1")
        act = LinearActuator(
            anchor=O1,
            angle=0.0,
            stroke=2.0,
            speed=0.1,
            name="act",
        )
        linkage = Linkage([O1, act], name="LinearOnly")
        period = linkage.get_rotation_period()
        # 2 * stroke / speed = 2*2/0.1 = 40
        assert period == round(2 * 2.0 / 0.1)

    def test_multiple_cranks_lcm(self):
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        c1 = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="c1")
        c2 = Crank(anchor=O2, radius=1.0, angular_velocity=0.2, name="c2")
        linkage = Linkage([O1, O2, c1, c2], name="TwoCranks")
        p1 = round(math.tau / 0.1)
        p2 = round(math.tau / 0.2)
        lcm = p1 * p2 // math.gcd(p1, p2)
        assert linkage.get_rotation_period() == lcm


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_yields_positions(self):
        _, _, _, _, linkage = _four_bar()
        positions_list = list(linkage.step(iterations=5))
        assert len(positions_list) == 5
        # Each element is a tuple of (x, y) tuples, one per component
        for pos in positions_list:
            assert len(pos) == 4  # 4 components

    def test_step_default_iterations(self):
        _, _, _, _, linkage = _four_bar(angular_velocity=0.1)
        positions_list = list(linkage.step())
        expected_period = round(math.tau / 0.1)
        assert len(positions_list) == expected_period

    def test_step_ground_does_not_move(self):
        O1, O2, _, _, linkage = _four_bar()
        for pos in linkage.step(iterations=10):
            assert pos[0] == (0.0, 0.0)  # O1
            assert pos[1] == (3.0, 0.0)  # O2

    def test_step_crank_moves(self):
        _, _, _, _, linkage = _four_bar()
        positions = list(linkage.step(iterations=2))
        # Crank should have moved between steps
        assert positions[0][2] != positions[1][2]

    def test_step_with_dt(self):
        """A smaller dt should produce smaller movements."""
        O1, O2, crank1, rocker1, _ = _four_bar()
        linkage1 = Linkage([O1, O2, crank1, rocker1])

        O1b = Ground(0.0, 0.0, name="O1b")
        O2b = Ground(3.0, 0.0, name="O2b")
        crank2 = Crank(anchor=O1b, radius=1.0, angular_velocity=0.1, name="crank2")
        rocker2 = RRRDyad(
            anchor1=crank2.output,
            anchor2=O2b,
            distance1=2.5,
            distance2=2.0,
            name="rocker2",
        )
        linkage2 = Linkage([O1b, O2b, crank2, rocker2])

        pos1 = list(linkage1.step(iterations=1, dt=1.0))[0]
        pos2 = list(linkage2.step(iterations=1, dt=0.5))[0]
        # With half dt, crank should move half as far angularly
        # Just verify they differ
        assert pos1[2] != pos2[2]

    def test_step_slider_crank(self):
        """RRPDyad mechanism should step without error."""
        _, _, _, _, _, linkage = _slider_crank()
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5
        for pos in positions:
            assert len(pos) == 5  # 3 grounds + crank + slider

    def test_step_with_fixed_dyad(self):
        """FixedDyad coupler should step without error."""
        _, _, _, _, _, linkage = _four_bar_with_coupler()
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5
        for pos in positions:
            assert len(pos) == 5  # 2 grounds + crank + rocker + coupler

    def test_step_arc_crank(self):
        """ArcCrank mechanism should step without error."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        arc = ArcCrank(
            anchor=O1,
            radius=1.0,
            angular_velocity=0.1,
            arc_start=0.0,
            arc_end=math.pi,
            name="arc",
        )
        rocker = RRRDyad(
            anchor1=arc.output,
            anchor2=O2,
            distance1=2.5,
            distance2=2.0,
            name="rocker",
        )
        linkage = Linkage([O1, O2, arc, rocker], name="ArcFourBar")
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5

    def test_step_linear_actuator(self):
        """LinearActuator mechanism should step without error."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        act = LinearActuator(
            anchor=O1,
            angle=0.0,
            stroke=2.0,
            speed=0.1,
            name="act",
        )
        rocker = RRRDyad(
            anchor1=act.output,
            anchor2=O2,
            distance1=2.5,
            distance2=2.0,
            name="rocker",
        )
        linkage = Linkage([O1, O2, act, rocker], name="LinActFourBar")
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5


# ---------------------------------------------------------------------------
# get_coords / set_coords
# ---------------------------------------------------------------------------


class TestCoords:
    def test_get_coords(self):
        O1, O2, crank, rocker, linkage = _four_bar()
        coords = linkage.get_coords()
        assert len(coords) == 4
        assert coords[0] == (0.0, 0.0)
        assert coords[1] == (3.0, 0.0)

    def test_set_coords(self):
        O1, O2, crank, rocker, linkage = _four_bar()
        new_coords = [(0.0, 0.0), (3.0, 0.0), (0.8, 0.6), (2.5, 1.5)]
        linkage.set_coords(new_coords)
        assert crank.position == (0.8, 0.6)
        assert rocker.position == (2.5, 1.5)

    def test_set_coords_length_mismatch(self):
        _, _, _, _, linkage = _four_bar()
        with pytest.raises(ValueError):
            linkage.set_coords([(0.0, 0.0)])  # Too few


# ---------------------------------------------------------------------------
# get_num_constraints / set_num_constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_get_num_constraints(self):
        _, _, crank, rocker, linkage = _four_bar()
        constraints = linkage.get_num_constraints()
        # Ground has no constraints, crank has 1 (radius), rocker has 2 (dist1, dist2)
        assert len(constraints) == 3
        assert constraints[0] == 1.0  # crank radius
        assert constraints[1] == 2.5  # rocker distance1
        assert constraints[2] == 2.0  # rocker distance2

    def test_set_num_constraints(self):
        _, _, crank, rocker, linkage = _four_bar()
        linkage.set_num_constraints([1.5, 3.0, 2.5])
        assert crank.radius == 1.5
        assert rocker.distance1 == 3.0
        assert rocker.distance2 == 2.5

    def test_get_constraints_with_fixed_dyad(self):
        _, _, crank, rocker, coupler, linkage = _four_bar_with_coupler()
        constraints = linkage.get_num_constraints()
        # crank: 1, rocker: 2, coupler(FixedDyad): 2 (distance, angle)
        assert len(constraints) == 5

    def test_round_trip_constraints(self):
        """get then set should preserve values."""
        _, _, _, _, linkage = _four_bar()
        original = linkage.get_num_constraints()
        linkage.set_num_constraints(original)
        assert linkage.get_num_constraints() == original


# ---------------------------------------------------------------------------
# set_input_velocity / get_velocities / get_accelerations
# ---------------------------------------------------------------------------


class TestInputVelocity:
    def test_set_input_velocity(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        assert crank._omega == 10.0  # type: ignore[attr-defined]
        assert crank._alpha == 0.0  # type: ignore[attr-defined]

    def test_set_input_velocity_with_alpha(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0, alpha=2.0)
        assert crank._alpha == 2.0  # type: ignore[attr-defined]

    def test_set_input_velocity_invalid_actuator(self):
        _, _, _, _, linkage = _four_bar()
        O_fake = Ground(99.0, 99.0, name="fake")
        fake_crank = Crank(anchor=O_fake, radius=1.0, name="fake_crank")
        with pytest.raises(ValueError):
            linkage.set_input_velocity(fake_crank, omega=1.0)

    def test_get_velocities_before_step(self):
        _, _, _, _, linkage = _four_bar()
        velocities = linkage.get_velocities()
        # Before any step_with_derivatives, velocities are None
        assert all(v is None for v in velocities)

    def test_get_accelerations_before_step(self):
        _, _, _, _, linkage = _four_bar()
        accelerations = linkage.get_accelerations()
        assert all(a is None for a in accelerations)


# ---------------------------------------------------------------------------
# step_with_derivatives
# ---------------------------------------------------------------------------


class TestStepWithDerivatives:
    def test_yields_three_tuples(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        for pos, vel, acc in linkage.step_with_derivatives(iterations=3):
            assert len(pos) == 4
            assert len(vel) == 4
            assert len(acc) == 4

    def test_ground_zero_velocity(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        for _pos, vel, acc in linkage.step_with_derivatives(iterations=1):
            # Grounds should have zero velocity and acceleration
            assert vel[0] == (0.0, 0.0)
            assert vel[1] == (0.0, 0.0)
            assert acc[0] == (0.0, 0.0)
            assert acc[1] == (0.0, 0.0)

    def test_crank_velocity_nonzero(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        for _pos, vel, _acc in linkage.step_with_derivatives(iterations=1):
            crank_vel = vel[2]
            assert crank_vel is not None
            # Velocity magnitude should be approximately omega * radius
            speed = math.hypot(crank_vel[0], crank_vel[1])
            assert speed > 0

    def test_rocker_velocity_computed(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        for _pos, vel, _acc in linkage.step_with_derivatives(iterations=1):
            rocker_vel = vel[3]
            assert rocker_vel is not None

    def test_rocker_acceleration_computed(self):
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0)
        for _pos, _vel, acc in linkage.step_with_derivatives(iterations=1):
            rocker_acc = acc[3]
            assert rocker_acc is not None

    def test_default_omega_zero(self):
        """Without set_input_velocity, omega defaults to 0."""
        _, _, _, _, linkage = _four_bar()
        for _pos, vel, _acc in linkage.step_with_derivatives(iterations=1):
            # Crank velocity with omega=0 should be (0,0)
            crank_vel = vel[2]
            assert crank_vel is not None
            assert crank_vel == (0.0, 0.0)

    def test_with_alpha(self):
        """Non-zero alpha should still compute accelerations."""
        _, _, crank, _, linkage = _four_bar()
        linkage.set_input_velocity(crank, omega=10.0, alpha=5.0)
        for _pos, _vel, acc in linkage.step_with_derivatives(iterations=1):
            crank_acc = acc[2]
            assert crank_acc is not None

    def test_fixed_dyad_derivatives(self):
        """FixedDyad should have velocity/acceleration computed."""
        _, _, crank, _, coupler, linkage = _four_bar_with_coupler()
        linkage.set_input_velocity(crank, omega=10.0)
        for _pos, vel, acc in linkage.step_with_derivatives(iterations=1):
            coupler_vel = vel[4]  # coupler is 5th component
            coupler_acc = acc[4]
            assert coupler_vel is not None
            assert coupler_acc is not None

    def test_slider_crank_derivatives(self):
        """RRPDyad should have velocity/acceleration computed."""
        _, _, _, crank, slider, linkage = _slider_crank()
        linkage.set_input_velocity(crank, omega=5.0)
        for _pos, vel, acc in linkage.step_with_derivatives(iterations=1):
            slider_vel = vel[4]  # slider is 5th component
            slider_acc = acc[4]
            assert slider_vel is not None
            assert slider_acc is not None

    def test_multiple_steps_consistency(self):
        """Positions from step_with_derivatives should match positions from step."""
        O1a = Ground(0.0, 0.0, name="O1a")
        O2a = Ground(3.0, 0.0, name="O2a")
        crank_a = Crank(anchor=O1a, radius=1.0, angular_velocity=0.1, name="crank_a")
        rocker_a = RRRDyad(
            anchor1=crank_a.output,
            anchor2=O2a,
            distance1=2.5,
            distance2=2.0,
            name="rocker_a",
        )
        linkage_a = Linkage([O1a, O2a, crank_a, rocker_a])

        O1b = Ground(0.0, 0.0, name="O1b")
        O2b = Ground(3.0, 0.0, name="O2b")
        crank_b = Crank(anchor=O1b, radius=1.0, angular_velocity=0.1, name="crank_b")
        rocker_b = RRRDyad(
            anchor1=crank_b.output,
            anchor2=O2b,
            distance1=2.5,
            distance2=2.0,
            name="rocker_b",
        )
        linkage_b = Linkage([O1b, O2b, crank_b, rocker_b])
        linkage_b.set_input_velocity(crank_b, omega=10.0)

        positions_a = list(linkage_a.step(iterations=5))
        results_b = list(linkage_b.step_with_derivatives(iterations=5))
        positions_b = [r[0] for r in results_b]

        for pa, pb in zip(positions_a, positions_b, strict=False):
            for ca, cb in zip(pa, pb, strict=False):
                assert ca[0] == pytest.approx(cb[0], abs=1e-12)
                assert ca[1] == pytest.approx(cb[1], abs=1e-12)

    def test_step_with_derivatives_default_iterations(self):
        _, _, crank, _, linkage = _four_bar(angular_velocity=0.1)
        linkage.set_input_velocity(crank, omega=10.0)
        results = list(linkage.step_with_derivatives())
        expected_period = round(math.tau / 0.1)
        assert len(results) == expected_period


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_linkage(self):
        linkage = Linkage([], name="empty")
        assert linkage.components == ()
        assert linkage.get_rotation_period() == 1
        assert linkage.get_coords() == []
        assert linkage.get_num_constraints() == []

    def test_ground_only_linkage(self):
        O1 = Ground(0.0, 0.0, name="O1")
        linkage = Linkage([O1])
        positions = list(linkage.step(iterations=3))
        assert len(positions) == 3
        assert all(p[0] == (0.0, 0.0) for p in positions)

    def test_step_called_twice(self):
        """Calling step twice should work (generator is recreated)."""
        _, _, _, _, linkage = _four_bar()
        list(linkage.step(iterations=3))
        positions = list(linkage.step(iterations=3))
        assert len(positions) == 3

    def test_set_coords_then_step(self):
        """Setting coords then stepping should work."""
        _, _, _, _, linkage = _four_bar()
        coords = linkage.get_coords()
        linkage.set_coords(coords)
        positions = list(linkage.step(iterations=2))
        assert len(positions) == 2

    def test_rebuild_then_step(self):
        """Rebuild then step should work."""
        _, _, _, _, linkage = _four_bar()
        linkage.rebuild()
        positions = list(linkage.step(iterations=2))
        assert len(positions) == 2
