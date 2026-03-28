"""Tests for the dyads module.

These tests verify the dyads API works correctly for building
and simulating planar linkage mechanisms.
"""

import math

import pytest

from pylinkage.dyads import (
    ArcCrank,
    Crank,
    FixedDyad,
    Ground,
    LinearActuator,
    Linkage,
    PointTracker,
    RRPDyad,
    RRRDyad,
)
from pylinkage.exceptions import UnbuildableError


class TestGround:
    """Tests for Ground dyad."""

    def test_ground_creation(self):
        """Test creating a ground point."""
        g = Ground(1.0, 2.0, name="O1")
        assert g.x == 1.0
        assert g.y == 2.0
        assert g.name == "O1"
        assert g.position == (1.0, 2.0)

    def test_ground_constraints(self):
        """Test that ground has no constraints."""
        g = Ground(0.0, 0.0)
        assert g.get_constraints() == ()

    def test_ground_reload_noop(self):
        """Test that reload doesn't change ground position."""
        g = Ground(1.0, 2.0)
        g.reload()
        assert g.position == (1.0, 2.0)


class TestCrank:
    """Tests for Crank dyad."""

    def test_crank_creation(self):
        """Test creating a crank."""
        origin = Ground(0.0, 0.0, name="O")
        crank = Crank(anchor=origin, radius=2.0, angular_velocity=0.1)

        assert crank.anchor == origin
        assert crank.radius == 2.0
        assert crank.angular_velocity == 0.1
        # Initial position at angle 0 should be (2, 0)
        assert crank.x == pytest.approx(2.0)
        assert crank.y == pytest.approx(0.0)

    def test_crank_with_initial_angle(self):
        """Test creating a crank with initial angle."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=1.0, initial_angle=math.pi / 2)

        # Initial position at 90 degrees should be (0, 1)
        assert crank.x == pytest.approx(0.0)
        assert crank.y == pytest.approx(1.0)

    def test_crank_output(self):
        """Test crank output proxy."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=1.0)

        assert crank.output.position == crank.position
        assert crank.output.x == crank.x
        assert crank.output.y == crank.y

    def test_crank_constraints(self):
        """Test crank constraints (radius)."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=2.0)

        assert crank.get_constraints() == (2.0,)

        crank.set_constraints(3.0)
        assert crank.radius == 3.0
        assert crank.get_constraints() == (3.0,)

    def test_crank_reload(self):
        """Test that crank rotates on reload."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=1.0, angular_velocity=math.pi / 2)

        # Initial position
        assert crank.x == pytest.approx(1.0)
        assert crank.y == pytest.approx(0.0)

        # After one step (90 degrees)
        crank.reload(dt=1.0)
        assert crank.x == pytest.approx(0.0)
        assert crank.y == pytest.approx(1.0)


class TestLinearActuator:
    """Tests for LinearActuator dyad."""

    def test_linear_actuator_creation(self):
        """Test creating a linear actuator."""
        origin = Ground(0.0, 0.0, name="O")
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.1, name="actuator")

        assert actuator.anchor == origin
        assert actuator.angle == 0.0
        assert actuator.stroke == 2.0
        assert actuator.speed == 0.1
        # Initial position at extension 0 should be at anchor
        assert actuator.x == pytest.approx(0.0)
        assert actuator.y == pytest.approx(0.0)

    def test_linear_actuator_with_angle(self):
        """Test creating a linear actuator with non-zero angle."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(
            anchor=origin,
            angle=math.pi / 2,  # 90 degrees (vertical)
            stroke=2.0,
            speed=0.1,
            initial_extension=1.0,
        )

        # Initial position at 90 degrees with extension 1 should be (0, 1)
        assert actuator.x == pytest.approx(0.0)
        assert actuator.y == pytest.approx(1.0)

    def test_linear_actuator_output(self):
        """Test linear actuator output proxy."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=2.0)

        assert actuator.output.position == actuator.position
        assert actuator.output.x == actuator.x
        assert actuator.output.y == actuator.y

    def test_linear_actuator_constraints(self):
        """Test linear actuator constraints (stroke, speed)."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.1)

        assert actuator.get_constraints() == (2.0, 0.1)

        actuator.set_constraints(3.0, 0.2)
        assert actuator.stroke == 3.0
        assert actuator.speed == 0.2
        assert actuator.get_constraints() == (3.0, 0.2)

    def test_linear_actuator_reload(self):
        """Test that linear actuator moves on reload."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.5)

        # Initial position at anchor
        assert actuator.x == pytest.approx(0.0)
        assert actuator.y == pytest.approx(0.0)
        assert actuator.extension == pytest.approx(0.0)

        # After one step (move 0.5 units along x-axis)
        actuator.reload(dt=1.0)
        assert actuator.x == pytest.approx(0.5)
        assert actuator.y == pytest.approx(0.0)
        assert actuator.extension == pytest.approx(0.5)

    def test_linear_actuator_oscillation(self):
        """Test that linear actuator oscillates at stroke limits."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(
            anchor=origin, angle=0.0, stroke=1.0, speed=0.6, initial_extension=0.0
        )

        # Move forward: 0 -> 0.6
        actuator.reload(dt=1.0)
        assert actuator.extension == pytest.approx(0.6)

        # Move forward: 0.6 -> 1.0, then bounce to 0.8 (overshoot by 0.2)
        actuator.reload(dt=1.0)
        assert actuator.extension == pytest.approx(0.8)
        assert actuator._direction == -1.0  # Now moving backward

        # Move backward: 0.8 -> 0.2
        actuator.reload(dt=1.0)
        assert actuator.extension == pytest.approx(0.2)

    def test_linear_actuator_with_initial_extension(self):
        """Test creating actuator with initial extension."""
        origin = Ground(1.0, 1.0)
        actuator = LinearActuator(
            anchor=origin, angle=0.0, stroke=3.0, speed=0.1, initial_extension=1.5
        )

        # Initial position should be anchor + (1.5, 0)
        assert actuator.x == pytest.approx(2.5)
        assert actuator.y == pytest.approx(1.0)
        assert actuator.extension == pytest.approx(1.5)

    def test_linear_actuator_invalid_stroke(self):
        """Test that invalid stroke raises error."""
        origin = Ground(0.0, 0.0)

        with pytest.raises(ValueError, match="Stroke must be positive"):
            LinearActuator(anchor=origin, angle=0.0, stroke=0.0, speed=0.1)

        with pytest.raises(ValueError, match="Stroke must be positive"):
            LinearActuator(anchor=origin, angle=0.0, stroke=-1.0, speed=0.1)

    def test_linear_actuator_invalid_initial_extension(self):
        """Test that invalid initial extension raises error."""
        origin = Ground(0.0, 0.0)

        with pytest.raises(ValueError, match="Initial extension"):
            LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.1, initial_extension=3.0)

        with pytest.raises(ValueError, match="Initial extension"):
            LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.1, initial_extension=-0.5)


class TestRRRDyad:
    """Tests for RRRDyad (circle-circle intersection)."""

    def test_rrr_creation(self):
        """Test creating an RRR dyad."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(4.0, 0.0, name="O2")

        dyad = RRRDyad(
            anchor1=origin1,
            anchor2=origin2,
            distance1=3.0,
            distance2=3.0,
            name="dyad",
        )

        # Should be at intersection of two circles
        # Circles: center (0,0) r=3, center (4,0) r=3
        # Intersection at (2, sqrt(5)) or (2, -sqrt(5))
        assert dyad.x == pytest.approx(2.0)
        assert abs(dyad.y) == pytest.approx(math.sqrt(5))

    def test_rrr_with_crank_output(self):
        """Test RRR dyad connected to crank output."""
        origin1 = Ground(0.0, 0.0)
        origin2 = Ground(4.0, 0.0)
        crank = Crank(anchor=origin1, radius=1.0)

        dyad = RRRDyad(
            anchor1=crank.output,
            anchor2=origin2,
            distance1=3.0,
            distance2=2.0,
        )

        assert dyad.x is not None
        assert dyad.y is not None

    def test_rrr_constraints(self):
        """Test RRR dyad constraints (two distances)."""
        origin1 = Ground(0.0, 0.0)
        origin2 = Ground(4.0, 0.0)

        dyad = RRRDyad(anchor1=origin1, anchor2=origin2, distance1=3.0, distance2=3.0)

        assert dyad.get_constraints() == (3.0, 3.0)

        dyad.set_constraints(2.5, 2.5)
        assert dyad.distance1 == 2.5
        assert dyad.distance2 == 2.5

    def test_rrr_unbuildable(self):
        """Test that unbuildable RRR raises error."""
        origin1 = Ground(0.0, 0.0)
        origin2 = Ground(10.0, 0.0)

        # Circles too far apart
        dyad = RRRDyad(anchor1=origin1, anchor2=origin2, distance1=1.0, distance2=1.0)

        with pytest.raises(UnbuildableError):
            dyad.reload()


class TestRRPDyad:
    """Tests for RRPDyad (circle-line intersection)."""

    def test_rrp_creation(self):
        """Test creating an RRP dyad."""
        origin = Ground(0.0, 1.0, name="O")
        L1 = Ground(0.0, 0.0, name="L1")
        L2 = Ground(4.0, 0.0, name="L2")

        dyad = RRPDyad(
            revolute_anchor=origin,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=1.0,
            name="slider",
        )

        # Should be on the line y=0 at distance 1 from (0,1)
        # That's at (0, 0) exactly
        assert dyad.y == pytest.approx(0.0)
        assert dyad.x is not None

    def test_rrp_constraints(self):
        """Test RRP dyad constraints (distance)."""
        origin = Ground(0.0, 1.0)
        L1 = Ground(0.0, 0.0)
        L2 = Ground(4.0, 0.0)

        dyad = RRPDyad(
            revolute_anchor=origin,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=1.0,
        )

        assert dyad.get_constraints() == (1.0,)

        dyad.set_constraints(1.5)
        assert dyad.distance == 1.5


class TestFixedDyad:
    """Tests for FixedDyad (polar projection)."""

    def test_fixed_creation(self):
        """Test creating a fixed dyad."""
        A = Ground(0.0, 0.0, name="A")
        B = Ground(1.0, 0.0, name="B")

        dyad = FixedDyad(
            anchor1=A,
            anchor2=B,
            distance=1.0,
            angle=math.pi / 2,
            name="fixed",
        )

        # Should be perpendicular to AB at distance 1 from A
        assert dyad.x == pytest.approx(0.0)
        assert dyad.y == pytest.approx(1.0)

    def test_fixed_constraints(self):
        """Test fixed dyad constraints (distance, angle)."""
        A = Ground(0.0, 0.0)
        B = Ground(1.0, 0.0)

        dyad = FixedDyad(anchor1=A, anchor2=B, distance=1.0, angle=0.0)

        assert dyad.get_constraints() == (1.0, 0.0)

        dyad.set_constraints(2.0, math.pi / 4)
        assert dyad.distance == 2.0
        assert dyad.angle == math.pi / 4


class TestLinkage:
    """Tests for Linkage container."""

    def test_linkage_creation(self):
        """Test creating a linkage."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=origin1, radius=1.0, angular_velocity=0.1)

        linkage = Linkage([origin1, origin2, crank], name="Test")

        assert linkage.name == "Test"
        assert len(linkage.dyads) == 3
        assert len(linkage._cranks) == 1

    def test_four_bar_simulation(self):
        """Test simulating a four-bar linkage."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=origin1, radius=0.5, angular_velocity=0.1)
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=origin2,
            distance1=1.5,
            distance2=1.0,
            name="rocker",
        )

        linkage = Linkage([origin1, origin2, crank, rocker], name="Four-Bar")

        # Run a few steps
        positions_list = list(linkage.step(iterations=5))

        assert len(positions_list) == 5
        for positions in positions_list:
            assert len(positions) == 4
            # Ground positions should not change
            assert positions[0] == (0.0, 0.0)
            assert positions[1] == (2.0, 0.0)

    def test_linkage_get_coords(self):
        """Test getting linkage coordinates."""
        origin1 = Ground(0.0, 0.0)
        origin2 = Ground(2.0, 0.0)

        linkage = Linkage([origin1, origin2])

        coords = linkage.get_coords()
        assert coords == [(0.0, 0.0), (2.0, 0.0)]

    def test_linkage_constraints(self):
        """Test getting/setting linkage constraints."""
        origin1 = Ground(0.0, 0.0)
        origin2 = Ground(2.0, 0.0)
        crank = Crank(anchor=origin1, radius=1.0)
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=origin2,
            distance1=2.0,
            distance2=1.5,
        )

        linkage = Linkage([origin1, origin2, crank, rocker])

        # Get constraints (crank radius + rocker distances)
        constraints = linkage.get_num_constraints()
        assert constraints == [1.0, 2.0, 1.5]

        # Set new constraints
        linkage.set_num_constraints([0.8, 2.2, 1.7])

        assert crank.radius == 0.8
        assert rocker.distance1 == 2.2
        assert rocker.distance2 == 1.7

    def test_linkage_rotation_period(self):
        """Test computing rotation period."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=1.0, angular_velocity=0.1)

        linkage = Linkage([origin, crank])

        # tau / 0.1 = ~63 steps per rotation
        period = linkage.get_rotation_period()
        assert period == round(math.tau / 0.1)

    def test_linkage_with_linear_actuator(self):
        """Test creating a linkage with linear actuator."""
        origin = Ground(0.0, 0.0, name="O")
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=2.0, speed=0.1)

        linkage = Linkage([origin, actuator], name="Test")

        assert len(linkage.dyads) == 2
        assert len(linkage._linear_actuators) == 1

    def test_linkage_linear_actuator_period(self):
        """Test computing period for linear actuator."""
        origin = Ground(0.0, 0.0)
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=1.0, speed=0.1)

        linkage = Linkage([origin, actuator])

        # Full cycle = 2 * stroke / speed = 2 * 1.0 / 0.1 = 20
        period = linkage.get_rotation_period()
        assert period == 20

    def test_linkage_mixed_actuators_period(self):
        """Test period with both crank and linear actuator."""
        origin = Ground(0.0, 0.0)
        crank = Crank(anchor=origin, radius=1.0, angular_velocity=math.pi / 5)
        actuator = LinearActuator(anchor=origin, angle=0.0, stroke=1.0, speed=0.2)

        linkage = Linkage([origin, crank, actuator])

        # Crank period: tau / (pi/5) = 10 steps
        # Actuator period: 2 * 1.0 / 0.2 = 10 steps
        # LCM(10, 10) = 10
        period = linkage.get_rotation_period()
        assert period == 10


class TestIntegration:
    """Integration tests for the dyads module."""

    def test_slider_crank_mechanism(self):
        """Test a slider-crank mechanism."""
        origin = Ground(0.0, 0.0, name="O")
        L1 = Ground(-3.0, 0.0, name="L1")  # Horizontal line through origin
        L2 = Ground(3.0, 0.0, name="L2")

        # Small crank with connecting rod longer than crank
        crank = Crank(anchor=origin, radius=0.5, angular_velocity=0.2)
        slider = RRPDyad(
            revolute_anchor=crank.output,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=2.0,  # Connecting rod > crank radius ensures buildable
            name="slider",
        )

        linkage = Linkage([origin, L1, L2, crank, slider], name="Slider-Crank")

        # Should simulate without error
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10

    def test_complex_linkage(self):
        """Test a more complex linkage with multiple dyads."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(3.0, 0.0, name="O2")

        crank = Crank(anchor=origin1, radius=0.8, angular_velocity=0.1)
        coupler = RRRDyad(
            anchor1=crank.output,
            anchor2=origin2,
            distance1=2.0,
            distance2=1.5,
            name="coupler",
        )
        # Add a fixed point on the coupler
        tracer = FixedDyad(
            anchor1=crank.output,
            anchor2=coupler,
            distance=1.0,
            angle=math.pi / 3,
            name="tracer",
        )

        linkage = Linkage([origin1, origin2, crank, coupler, tracer], name="Complex")

        # Should simulate without error
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5

    def test_linear_actuator_driven_mechanism(self):
        """Test mechanism driven by linear actuator."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(3.0, 0.0, name="O2")

        # Linear actuator driving a four-bar-like mechanism
        actuator = LinearActuator(
            anchor=origin1,
            angle=math.pi / 4,  # 45 degrees
            stroke=1.5,
            speed=0.1,
            initial_extension=0.75,  # Start in middle
        )

        # Connect a rocker to the actuator output
        rocker = RRRDyad(
            anchor1=actuator.output,
            anchor2=origin2,
            distance1=2.0,
            distance2=1.5,
            name="rocker",
        )

        linkage = Linkage([origin1, origin2, actuator, rocker], name="Actuator-Driven")

        # Should simulate without error
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10

        # Actuator output should have moved
        assert positions[-1][2] != positions[0][2]


class TestArcCrank:
    """Tests for ArcCrank (oscillating rotary input)."""

    def test_arc_crank_creation(self):
        """Test creating an arc crank."""
        origin = Ground(0.0, 0.0, name="O")
        arc_crank = ArcCrank(
            anchor=origin,
            radius=2.0,
            angular_velocity=0.1,
            arc_start=0.0,
            arc_end=math.pi / 2,
        )

        assert arc_crank.anchor == origin
        assert arc_crank.radius == 2.0
        assert arc_crank.angular_velocity == 0.1
        assert arc_crank.arc_start == 0.0
        assert arc_crank.arc_end == math.pi / 2
        # Initial position at arc_start (angle 0) should be (2, 0)
        assert arc_crank.x == pytest.approx(2.0)
        assert arc_crank.y == pytest.approx(0.0)

    def test_arc_crank_with_initial_angle(self):
        """Test creating an arc crank with custom initial angle."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(
            anchor=origin,
            radius=1.0,
            arc_start=0.0,
            arc_end=math.pi,
            initial_angle=math.pi / 2,
        )

        # Initial position at 90 degrees should be (0, 1)
        assert arc_crank.x == pytest.approx(0.0)
        assert arc_crank.y == pytest.approx(1.0)
        assert arc_crank.angle == pytest.approx(math.pi / 2)

    def test_arc_crank_output(self):
        """Test arc crank output proxy."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(anchor=origin, radius=1.0, arc_start=0, arc_end=math.pi)

        assert arc_crank.output.position == arc_crank.position
        assert arc_crank.output.x == arc_crank.x
        assert arc_crank.output.y == arc_crank.y

    def test_arc_crank_constraints(self):
        """Test arc crank constraints (radius, arc_start, arc_end)."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(anchor=origin, radius=2.0, arc_start=0.0, arc_end=math.pi / 2)

        assert arc_crank.get_constraints() == (2.0, 0.0, math.pi / 2)

        arc_crank.set_constraints(3.0, math.pi / 4, math.pi)
        assert arc_crank.radius == 3.0
        assert arc_crank.arc_start == math.pi / 4
        assert arc_crank.arc_end == math.pi

    def test_arc_crank_reload(self):
        """Test that arc crank rotates on reload."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(
            anchor=origin,
            radius=1.0,
            angular_velocity=math.pi / 4,  # 45 degrees per step
            arc_start=0.0,
            arc_end=math.pi,
        )

        # Initial position
        assert arc_crank.x == pytest.approx(1.0)
        assert arc_crank.y == pytest.approx(0.0)
        assert arc_crank.angle == pytest.approx(0.0)

        # After one step (45 degrees)
        arc_crank.reload(dt=1.0)
        assert arc_crank.x == pytest.approx(math.cos(math.pi / 4))
        assert arc_crank.y == pytest.approx(math.sin(math.pi / 4))
        assert arc_crank.angle == pytest.approx(math.pi / 4)

    def test_arc_crank_bounce_at_end(self):
        """Test that arc crank bounces at arc_end."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(
            anchor=origin,
            radius=1.0,
            angular_velocity=0.6,  # More than half of arc range
            arc_start=0.0,
            arc_end=1.0,  # 1 radian arc
        )

        # Start at arc_start
        assert arc_crank.angle == pytest.approx(0.0)

        # After one step, should hit arc_end and bounce back
        arc_crank.reload(dt=1.0)
        assert arc_crank.angle == pytest.approx(0.6)

        # After second step, should overshoot arc_end and bounce
        arc_crank.reload(dt=1.0)
        # 0.6 + 0.6 = 1.2, overshoots by 0.2, so bounces to 1.0 - 0.2 = 0.8
        assert arc_crank.angle == pytest.approx(0.8)

    def test_arc_crank_bounce_at_start(self):
        """Test that arc crank bounces at arc_start."""
        origin = Ground(0.0, 0.0)
        arc_crank = ArcCrank(
            anchor=origin,
            radius=1.0,
            angular_velocity=0.6,
            arc_start=0.0,
            arc_end=1.0,
            initial_angle=0.2,  # Start near arc_start
        )

        # Manually set direction to negative (moving toward arc_start)
        arc_crank._direction = -1.0

        # After one step, should hit arc_start and bounce
        arc_crank.reload(dt=1.0)
        # 0.2 - 0.6 = -0.4, undershoots by 0.4, so bounces to 0.0 + 0.4 = 0.4
        assert arc_crank.angle == pytest.approx(0.4)

    def test_arc_crank_invalid_arc_bounds(self):
        """Test that invalid arc bounds raise ValueError."""
        origin = Ground(0.0, 0.0)

        with pytest.raises(ValueError, match="arc_end must be greater than arc_start"):
            ArcCrank(
                anchor=origin,
                radius=1.0,
                arc_start=math.pi,  # arc_start > arc_end
                arc_end=0.0,
            )

    def test_arc_crank_invalid_initial_angle(self):
        """Test that initial_angle out of range raises ValueError."""
        origin = Ground(0.0, 0.0)

        with pytest.raises(ValueError, match="initial_angle must be between"):
            ArcCrank(
                anchor=origin,
                radius=1.0,
                arc_start=0.0,
                arc_end=math.pi / 2,
                initial_angle=math.pi,  # Outside arc range
            )

    def test_arc_crank_in_linkage(self):
        """Test arc crank in a linkage simulation."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(2.0, 0.0, name="O2")
        arc_crank = ArcCrank(
            anchor=origin1,
            radius=0.5,
            angular_velocity=0.1,
            arc_start=0.0,
            arc_end=math.pi / 2,
        )
        rocker = RRRDyad(
            anchor1=arc_crank.output,
            anchor2=origin2,
            distance1=1.5,
            distance2=1.0,
            name="rocker",
        )

        linkage = Linkage(
            [origin1, origin2, arc_crank, rocker], name="ArcCrank-Four-Bar"
        )

        # Check that arc crank is properly tracked
        assert len(linkage._arc_cranks) == 1

        # Period should be based on arc range
        period = linkage.get_rotation_period()
        expected_period = round(2 * (math.pi / 2) / 0.1)  # 2 * arc_range / velocity
        assert period == expected_period

        # Should simulate without error
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10


class TestPointTracker:
    """Tests for PointTracker (sensor component)."""

    def test_point_tracker_creation(self):
        """Test creating a point tracker."""
        A = Ground(0.0, 0.0, name="A")
        B = Ground(1.0, 0.0, name="B")

        tracker = PointTracker(
            anchor1=A,
            anchor2=B,
            distance=1.0,
            angle=math.pi / 2,
            name="tracker",
        )

        # Should be perpendicular to AB at distance 1 from A
        assert tracker.x == pytest.approx(0.0)
        assert tracker.y == pytest.approx(1.0)

    def test_point_tracker_no_constraints(self):
        """Test that point tracker has no optimizable constraints."""
        A = Ground(0.0, 0.0)
        B = Ground(1.0, 0.0)

        tracker = PointTracker(anchor1=A, anchor2=B, distance=1.0, angle=0.0)

        # Should return empty tuple - no constraints for optimization
        assert tracker.get_constraints() == ()

        # set_constraints should be a no-op
        tracker.set_constraints(2.0, math.pi / 4)
        # Distance and angle should remain unchanged
        assert tracker.distance == 1.0
        assert tracker.angle == 0.0

    def test_point_tracker_with_crank_output(self):
        """Test point tracker connected to crank output."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=origin1, radius=1.0, angular_velocity=0.1)

        tracker = PointTracker(
            anchor1=crank.output,
            anchor2=origin2,
            distance=0.5,
            angle=math.pi / 4,
            name="tracker",
        )

        # Position should be computed from crank output
        assert tracker.x is not None
        assert tracker.y is not None

    def test_point_tracker_in_linkage(self):
        """Test point tracker in a linkage simulation."""
        origin1 = Ground(0.0, 0.0, name="O1")
        origin2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=origin1, radius=0.5, angular_velocity=0.1)
        coupler = RRRDyad(
            anchor1=crank.output,
            anchor2=origin2,
            distance1=1.5,
            distance2=1.0,
            name="coupler",
        )
        tracker = PointTracker(
            anchor1=crank.output,
            anchor2=coupler,
            distance=0.5,
            angle=math.pi / 4,
            name="tracker",
        )

        linkage = Linkage(
            [origin1, origin2, crank, coupler, tracker], name="Tracked-Four-Bar"
        )

        # Tracker should not contribute to constraints
        constraints = linkage.get_num_constraints()
        # Should only have crank radius and coupler distances
        assert len(constraints) == 3  # crank.radius, coupler.distance1, coupler.distance2

        # Should simulate without error
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5

        # Tracker position should change as crank rotates
        assert positions[-1][4] != positions[0][4]

    def test_point_tracker_vs_fixed_dyad(self):
        """Test that PointTracker and FixedDyad compute same positions.

        They should have different constraints.
        """
        A = Ground(0.0, 0.0)
        B = Ground(1.0, 0.0)

        # Same parameters for both
        distance = 1.5
        angle = math.pi / 3

        tracker = PointTracker(anchor1=A, anchor2=B, distance=distance, angle=angle)
        fixed = FixedDyad(anchor1=A, anchor2=B, distance=distance, angle=angle)

        # Should compute the same position
        assert tracker.x == pytest.approx(fixed.x)
        assert tracker.y == pytest.approx(fixed.y)

        # But tracker has no constraints, fixed has constraints
        assert tracker.get_constraints() == ()
        assert fixed.get_constraints() == (distance, angle)
