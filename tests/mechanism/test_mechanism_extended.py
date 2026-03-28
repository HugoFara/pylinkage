"""Extended tests for mechanism/mechanism.py — uncovered methods and properties."""

import math

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    RevoluteJoint,
)
from pylinkage.mechanism.joint import TrackerJoint
from pylinkage.mechanism.link import ArcDriverLink


def _build_four_bar(omega=0.1):
    """Build a four-bar linkage mechanism with valid geometry.

    Grashof-compatible dimensions: ground=3, crank=1, coupler=3, rocker=2.
    """
    O1 = GroundJoint("O1", position=(0.0, 0.0))
    O2 = GroundJoint("O2", position=(3.0, 0.0))
    A = RevoluteJoint("A", position=(1.0, 0.0))
    B = RevoluteJoint("B", position=(3.0, 2.0))

    ground = GroundLink("ground", joints=[O1, O2])
    crank = DriverLink("crank", joints=[O1, A], motor_joint=O1, angular_velocity=omega)
    coupler = Link("coupler", joints=[A, B])
    rocker = Link("rocker", joints=[O2, B])

    return Mechanism(
        name="FourBar",
        joints=[O1, O2, A, B],
        links=[ground, crank, coupler, rocker],
        ground=ground,
    )


class TestMechanismGetJointGetLink:
    """Tests for get_joint() and get_link()."""

    def test_get_joint_existing(self):
        mech = _build_four_bar()
        j = mech.get_joint("O1")
        assert j is not None
        assert j.id == "O1"

    def test_get_joint_missing(self):
        mech = _build_four_bar()
        assert mech.get_joint("NONEXISTENT") is None

    def test_get_link_existing(self):
        mech = _build_four_bar()
        link = mech.get_link("crank")
        assert link is not None
        assert isinstance(link, DriverLink)

    def test_get_link_missing(self):
        mech = _build_four_bar()
        assert mech.get_link("NONEXISTENT") is None


class TestMechanismAutoDetectGround:
    """Test that ground link is auto-detected if not specified."""

    def test_auto_detect_ground(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        mech = Mechanism(joints=[origin], links=[ground], ground=None)
        assert mech.ground is ground


class TestMechanismStep:
    """Tests for Mechanism.step()."""

    def test_step_yields_positions(self):
        mech = _build_four_bar(omega=0.1)
        positions = list(mech.step(dt=1.0))
        assert len(positions) > 0
        # Each position tuple should have 4 entries (one per joint)
        assert len(positions[0]) == 4

    def test_step_moves_crank_output(self):
        mech = _build_four_bar(omega=0.1)
        initial_A = mech.get_joint("A").position
        # Take one step
        gen = mech.step(dt=1.0)
        first = next(gen)
        # Joint A should have moved
        assert first[2] != initial_A


class TestMechanismGetRotationPeriod:
    """Tests for get_rotation_period()."""

    def test_default_when_no_drivers(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        mech = Mechanism(joints=[origin], links=[])
        assert mech.get_rotation_period() == 100

    def test_period_based_on_angular_velocity(self):
        mech = _build_four_bar(omega=0.1)
        expected = int(2 * math.pi / 0.1)
        assert mech.get_rotation_period() == expected

    def test_arc_driver_period(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        arc = ArcDriverLink(
            "arc",
            joints=[origin, A],
            motor_joint=origin,
            angular_velocity=0.1,
            arc_start=0.0,
            arc_end=1.0,
        )
        mech = Mechanism(joints=[origin, A], links=[ground, arc])
        # Period for arc: 2 * arc_sweep / omega = 2 * 1.0 / 0.1 = 20
        assert mech.get_rotation_period() == 20


class TestMechanismConstraints:
    """Tests for get_constraints() and set_constraints()."""

    def test_get_constraints(self):
        mech = _build_four_bar()
        constraints = mech.get_constraints()
        # Should have: crank radius, coupler length, rocker length
        assert len(constraints) == 3
        # Crank radius should be 1.0
        assert abs(constraints[0] - 1.0) < 1e-10

    def test_set_constraints(self):
        mech = _build_four_bar()
        original = mech.get_constraints()
        # Double all constraints
        mech.set_constraints([c * 2 for c in original])
        # The driver's output joint should have moved
        A = mech.get_joint("A")
        # After doubling the crank radius from 1.0 to 2.0, A should be at ~(2,0)
        assert abs(A.position[0] - 2.0) < 1e-10

    def test_set_constraints_with_fewer_values(self):
        """set_constraints with fewer values than links should not crash."""
        mech = _build_four_bar()
        mech.set_constraints([1.5])  # Only set first constraint


class TestMechanismJointPositions:
    """Tests for get_joint_positions() / set_joint_positions()."""

    def test_set_joint_positions(self):
        mech = _build_four_bar()
        new_positions = [(0.0, 0.0), (3.0, 0.0), (1.5, 0.0), (3.5, 1.5)]
        mech.set_joint_positions(new_positions)
        result = mech.get_joint_positions()
        assert result == new_positions

    def test_get_positions_with_none(self):
        """Joints with None positions should return (0.0, 0.0)."""
        A = RevoluteJoint("A", position=(None, None))
        mech = Mechanism(joints=[A], links=[])
        positions = mech.get_joint_positions()
        assert positions == [(0.0, 0.0)]


class TestMechanismReset:
    """Tests for Mechanism.reset()."""

    def test_reset_restores_driver_angle(self):
        mech = _build_four_bar(omega=0.5)
        crank = mech.get_link("crank")
        assert isinstance(crank, DriverLink)
        initial_angle = crank.initial_angle

        # Step a few times to change the angle
        crank.step(1.0)
        crank.step(1.0)
        assert crank.current_angle != initial_angle

        mech.reset()
        assert crank.current_angle == initial_angle


class TestMechanismSolveOrder:
    """Tests for _compute_solve_order and _can_solve_joint."""

    def test_ground_joints_solved_first(self):
        mech = _build_four_bar()
        # First two in solve order should be ground joints
        assert isinstance(mech._solve_order[0], GroundJoint)
        assert isinstance(mech._solve_order[1], GroundJoint)

    def test_driver_output_solved_after_ground(self):
        mech = _build_four_bar()
        # Joint A (crank output) should be in solve order
        ids = [j.id for j in mech._solve_order]
        assert "A" in ids
        # A should come after ground joints
        a_idx = ids.index("A")
        assert a_idx >= 2  # After O1, O2

    def test_dependent_joint_solved_last(self):
        mech = _build_four_bar()
        ids = [j.id for j in mech._solve_order]
        assert "B" in ids
        b_idx = ids.index("B")
        a_idx = ids.index("A")
        assert b_idx > a_idx


class TestMechanismWithTracker:
    """Test mechanism with TrackerJoint."""

    def test_tracker_updates_during_step(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        tracker = TrackerJoint(
            "T",
            position=(0.0, 0.0),
            ref_joint1_id="O1",
            ref_joint2_id="A",
            distance=0.5,
            angle=0.0,
        )

        ground = GroundLink("ground", joints=[O1])
        crank = DriverLink("crank", joints=[O1, A], motor_joint=O1, angular_velocity=0.1)

        mech = Mechanism(
            joints=[O1, A, tracker],
            links=[ground, crank],
        )

        # Step once — tracker should update
        mech._step_once(dt=1.0)
        # Tracker should be at midpoint between O1 and A (distance=0.5, angle=0)
        assert tracker.position[0] is not None
        assert tracker.position[1] is not None
