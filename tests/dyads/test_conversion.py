"""Tests for dyads/_conversion.py (Linkage -> Mechanism conversion)."""

import pytest

from pylinkage.dyads import (
    Crank,
    FixedDyad,
    Ground,
    Linkage,
    RRPDyad,
    RRRDyad,
)
from pylinkage.dyads._conversion import to_mechanism
from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Mechanism,
)


def _make_fourbar() -> Linkage:
    """Build a simple four-bar using the dyads API."""
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    A = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, initial_angle=0.0, name="A")
    B = RRRDyad(anchor1=A, anchor2=O2, distance1=3.0, distance2=2.0, name="B")
    return Linkage(components=[O1, O2, A, B], name="Four-bar")


class TestToMechanismBasic:
    """Basic tests for to_mechanism()."""

    def test_returns_mechanism(self):
        """to_mechanism returns a Mechanism instance."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        assert isinstance(mech, Mechanism)

    def test_name_preserved(self):
        """Linkage name is preserved."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        assert mech.name == "Four-bar"

    def test_joint_count(self):
        """Each component becomes a joint."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        # O1, O2, A, B -> 4 joints
        assert len(mech.joints) == 4

    def test_ground_joints(self):
        """Ground components become GroundJoint instances."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        ground_joints = [j for j in mech.joints if isinstance(j, GroundJoint)]
        ground_names = {j.name for j in ground_joints}
        assert "O1" in ground_names
        assert "O2" in ground_names

    def test_ground_link_created(self):
        """A GroundLink is created from ground joints."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        ground_links = [lk for lk in mech.links if isinstance(lk, GroundLink)]
        assert len(ground_links) == 1
        assert len(ground_links[0].joints) == 2

    def test_driver_link_created(self):
        """A DriverLink is created for the Crank."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        driver_links = [lk for lk in mech.links if isinstance(lk, DriverLink)]
        assert len(driver_links) == 1
        dl = driver_links[0]
        assert dl.angular_velocity == pytest.approx(0.1)

    def test_rrr_creates_two_links(self):
        """RRRDyad creates two links connecting to its anchors."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        plain_links = [lk for lk in mech.links if not isinstance(lk, (GroundLink, DriverLink))]
        # RRRDyad B -> link1 (A-B) and link2 (O2-B)
        assert len(plain_links) == 2

    def test_ground_positions(self):
        """Ground joint positions are correct."""
        linkage = _make_fourbar()
        mech = to_mechanism(linkage)
        jmap = {j.name: j for j in mech.joints}
        assert jmap["O1"].position == pytest.approx((0.0, 0.0))
        assert jmap["O2"].position == pytest.approx((3.0, 0.0))


class TestToMechanismWithFixedDyad:
    """Tests with FixedDyad."""

    def test_fixed_dyad_creates_two_links(self):
        """FixedDyad creates two links connecting to its anchors."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        F = FixedDyad(anchor1=O1, anchor2=O2, distance=2.0, angle=0.5, name="F")
        linkage = Linkage(components=[O1, O2, F], name="Fixed")

        mech = to_mechanism(linkage)
        plain_links = [lk for lk in mech.links if not isinstance(lk, (GroundLink, DriverLink))]
        assert len(plain_links) == 2

    def test_fixed_dyad_joint_count(self):
        """FixedDyad adds one joint."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        F = FixedDyad(anchor1=O1, anchor2=O2, distance=2.0, angle=0.5, name="F")
        linkage = Linkage(components=[O1, O2, F], name="Fixed")

        mech = to_mechanism(linkage)
        assert len(mech.joints) == 3  # O1, O2, F


class TestToMechanismWithRRP:
    """Tests with RRPDyad (slider)."""

    def test_rrp_creates_two_links(self):
        """RRPDyad creates a link to revolute anchor and a guide link."""
        O1 = Ground(0.0, 0.0, name="O1")
        L1 = Ground(0.0, 2.0, name="L1")
        L2 = Ground(5.0, 2.0, name="L2")
        A = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="A")
        S = RRPDyad(
            revolute_anchor=A,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=2.5,
            name="S",
        )
        linkage = Linkage(components=[O1, L1, L2, A, S], name="Slider-Crank")

        mech = to_mechanism(linkage)

        # RRPDyad should produce: link1 (rev anchor - S) and guide (L1-L2-S)
        plain_links = [lk for lk in mech.links if not isinstance(lk, (GroundLink, DriverLink))]
        assert len(plain_links) == 2

        # The guide link should have 3 joints
        guide_links = [lk for lk in plain_links if len(lk.joints) == 3]
        assert len(guide_links) == 1

    def test_rrp_joint_count(self):
        """Slider-crank has 5 joints (O1, L1, L2, A, S)."""
        O1 = Ground(0.0, 0.0, name="O1")
        L1 = Ground(0.0, 2.0, name="L1")
        L2 = Ground(5.0, 2.0, name="L2")
        A = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="A")
        S = RRPDyad(
            revolute_anchor=A,
            line_anchor1=L1,
            line_anchor2=L2,
            distance=2.5,
            name="S",
        )
        linkage = Linkage(components=[O1, L1, L2, A, S], name="Slider-Crank")

        mech = to_mechanism(linkage)
        assert len(mech.joints) == 5


class TestToMechanismEdgeCases:
    """Edge case tests."""

    def test_ground_only(self):
        """Linkage with only ground produces GroundLink and no others."""
        O1 = Ground(0.0, 0.0, name="O1")
        linkage = Linkage(components=[O1], name="Ground-only")
        mech = to_mechanism(linkage)

        assert len(mech.joints) == 1
        assert len([lk for lk in mech.links if isinstance(lk, GroundLink)]) == 1

    def test_two_grounds(self):
        """Two grounds produce a single GroundLink with two joints."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(3.0, 0.0, name="O2")
        linkage = Linkage(components=[O1, O2], name="Two-grounds")
        mech = to_mechanism(linkage)

        ground_links = [lk for lk in mech.links if isinstance(lk, GroundLink)]
        assert len(ground_links) == 1
        assert len(ground_links[0].joints) == 2

    def test_crank_only(self):
        """Ground + Crank produces ground link and driver link."""
        O1 = Ground(0.0, 0.0, name="O1")
        A = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="A")
        linkage = Linkage(components=[O1, A], name="Crank-only")
        mech = to_mechanism(linkage)

        assert len(mech.joints) == 2
        assert len([lk for lk in mech.links if isinstance(lk, DriverLink)]) == 1
        assert len([lk for lk in mech.links if isinstance(lk, GroundLink)]) == 1
