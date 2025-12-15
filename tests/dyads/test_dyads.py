"""Tests for the dyads module.

These tests verify the dyads API works correctly for building
and simulating planar linkage mechanisms.
"""

import math

import pytest

from pylinkage.dyads import (
    Crank,
    FixedDyad,
    Ground,
    Linkage,
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
        O = Ground(0.0, 0.0, name="O")
        crank = Crank(anchor=O, radius=2.0, angular_velocity=0.1)

        assert crank.anchor == O
        assert crank.radius == 2.0
        assert crank.angular_velocity == 0.1
        # Initial position at angle 0 should be (2, 0)
        assert crank.x == pytest.approx(2.0)
        assert crank.y == pytest.approx(0.0)

    def test_crank_with_initial_angle(self):
        """Test creating a crank with initial angle."""
        O = Ground(0.0, 0.0)
        crank = Crank(anchor=O, radius=1.0, initial_angle=math.pi / 2)

        # Initial position at 90 degrees should be (0, 1)
        assert crank.x == pytest.approx(0.0)
        assert crank.y == pytest.approx(1.0)

    def test_crank_output(self):
        """Test crank output proxy."""
        O = Ground(0.0, 0.0)
        crank = Crank(anchor=O, radius=1.0)

        assert crank.output.position == crank.position
        assert crank.output.x == crank.x
        assert crank.output.y == crank.y

    def test_crank_constraints(self):
        """Test crank constraints (radius)."""
        O = Ground(0.0, 0.0)
        crank = Crank(anchor=O, radius=2.0)

        assert crank.get_constraints() == (2.0,)

        crank.set_constraints(3.0)
        assert crank.radius == 3.0
        assert crank.get_constraints() == (3.0,)

    def test_crank_reload(self):
        """Test that crank rotates on reload."""
        O = Ground(0.0, 0.0)
        crank = Crank(anchor=O, radius=1.0, angular_velocity=math.pi / 2)

        # Initial position
        assert crank.x == pytest.approx(1.0)
        assert crank.y == pytest.approx(0.0)

        # After one step (90 degrees)
        crank.reload(dt=1.0)
        assert crank.x == pytest.approx(0.0)
        assert crank.y == pytest.approx(1.0)


class TestRRRDyad:
    """Tests for RRRDyad (circle-circle intersection)."""

    def test_rrr_creation(self):
        """Test creating an RRR dyad."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(4.0, 0.0, name="O2")

        dyad = RRRDyad(
            anchor1=O1,
            anchor2=O2,
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
        O1 = Ground(0.0, 0.0)
        O2 = Ground(4.0, 0.0)
        crank = Crank(anchor=O1, radius=1.0)

        dyad = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
            distance1=3.0,
            distance2=2.0,
        )

        assert dyad.x is not None
        assert dyad.y is not None

    def test_rrr_constraints(self):
        """Test RRR dyad constraints (two distances)."""
        O1 = Ground(0.0, 0.0)
        O2 = Ground(4.0, 0.0)

        dyad = RRRDyad(anchor1=O1, anchor2=O2, distance1=3.0, distance2=3.0)

        assert dyad.get_constraints() == (3.0, 3.0)

        dyad.set_constraints(2.5, 2.5)
        assert dyad.distance1 == 2.5
        assert dyad.distance2 == 2.5

    def test_rrr_unbuildable(self):
        """Test that unbuildable RRR raises error."""
        O1 = Ground(0.0, 0.0)
        O2 = Ground(10.0, 0.0)

        # Circles too far apart
        dyad = RRRDyad(anchor1=O1, anchor2=O2, distance1=1.0, distance2=1.0)

        with pytest.raises(UnbuildableError):
            dyad.reload()


class TestRRPDyad:
    """Tests for RRPDyad (circle-line intersection)."""

    def test_rrp_creation(self):
        """Test creating an RRP dyad."""
        O = Ground(0.0, 1.0, name="O")
        L1 = Ground(0.0, 0.0, name="L1")
        L2 = Ground(4.0, 0.0, name="L2")

        dyad = RRPDyad(
            revolute_anchor=O,
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
        O = Ground(0.0, 1.0)
        L1 = Ground(0.0, 0.0)
        L2 = Ground(4.0, 0.0)

        dyad = RRPDyad(
            revolute_anchor=O,
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
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)

        linkage = Linkage([O1, O2, crank], name="Test")

        assert linkage.name == "Test"
        assert len(linkage.dyads) == 3
        assert len(linkage._cranks) == 1

    def test_four_bar_simulation(self):
        """Test simulating a four-bar linkage."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=O1, radius=0.5, angular_velocity=0.1)
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
            distance1=1.5,
            distance2=1.0,
            name="rocker",
        )

        linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")

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
        O1 = Ground(0.0, 0.0)
        O2 = Ground(2.0, 0.0)

        linkage = Linkage([O1, O2])

        coords = linkage.get_coords()
        assert coords == [(0.0, 0.0), (2.0, 0.0)]

    def test_linkage_constraints(self):
        """Test getting/setting linkage constraints."""
        O1 = Ground(0.0, 0.0)
        O2 = Ground(2.0, 0.0)
        crank = Crank(anchor=O1, radius=1.0)
        rocker = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
            distance1=2.0,
            distance2=1.5,
        )

        linkage = Linkage([O1, O2, crank, rocker])

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
        O = Ground(0.0, 0.0)
        crank = Crank(anchor=O, radius=1.0, angular_velocity=0.1)

        linkage = Linkage([O, crank])

        # tau / 0.1 = ~63 steps per rotation
        period = linkage.get_rotation_period()
        assert period == round(math.tau / 0.1)


class TestIntegration:
    """Integration tests for the dyads module."""

    def test_slider_crank_mechanism(self):
        """Test a slider-crank mechanism."""
        O = Ground(0.0, 0.0, name="O")
        L1 = Ground(-3.0, 0.0, name="L1")  # Horizontal line through origin
        L2 = Ground(3.0, 0.0, name="L2")

        # Small crank with connecting rod longer than crank
        crank = Crank(anchor=O, radius=0.5, angular_velocity=0.2)
        slider = RRPDyad(
            revolute_anchor=crank.output,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=2.0,  # Connecting rod > crank radius ensures buildable
            name="slider",
        )

        linkage = Linkage([O, L1, L2, crank, slider], name="Slider-Crank")

        # Should simulate without error
        positions = list(linkage.step(iterations=10))
        assert len(positions) == 10

    def test_complex_linkage(self):
        """Test a more complex linkage with multiple dyads."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")

        crank = Crank(anchor=O1, radius=0.8, angular_velocity=0.1)
        coupler = RRRDyad(
            anchor1=crank.output,
            anchor2=O2,
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

        linkage = Linkage([O1, O2, crank, coupler, tracer], name="Complex")

        # Should simulate without error
        positions = list(linkage.step(iterations=5))
        assert len(positions) == 5
