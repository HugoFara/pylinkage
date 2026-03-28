"""Tests for MechanismBuilder.

These tests verify the links-first builder approach for creating mechanisms.
"""

import math

import pytest

from pylinkage.exceptions import UnbuildableError, UnderconstrainedError
from pylinkage.mechanism import Mechanism, MechanismBuilder


class TestMechanismBuilderBasic:
    """Basic tests for MechanismBuilder."""

    def test_builder_creation(self):
        """Test creating a builder."""
        builder = MechanismBuilder("test")
        assert builder.name == "test"

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        builder = MechanismBuilder("test")
        result = builder.add_ground_link("ground", ports={"O1": (0, 0)})
        assert result is builder


class TestFourBarAssembly:
    """Tests for four-bar linkage assembly."""

    def test_simple_fourbar(self):
        """Test assembling a simple four-bar linkage."""
        mechanism = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .build()
        )

        assert isinstance(mechanism, Mechanism)
        assert mechanism.name == "four-bar"

        # Should have 4 joints
        assert len(mechanism.joints) == 4

        # Should have 4 links (ground + crank + coupler + rocker)
        assert len(mechanism.links) == 4

    def test_fourbar_with_initial_angle(self):
        """Test four-bar with non-zero initial crank angle."""
        mechanism = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link(
                "crank", length=1.0, motor_port="O1", omega=0.1, initial_angle=math.pi / 4
            )
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .build()
        )

        # Verify crank tip is at expected position
        crank_tip = None
        for joint in mechanism.joints:
            if "crank" in joint.id and "tip" in joint.id:
                crank_tip = joint
                break

        assert crank_tip is not None
        expected_x = math.cos(math.pi / 4)
        expected_y = math.sin(math.pi / 4)
        assert abs(crank_tip.x - expected_x) < 0.01
        assert abs(crank_tip.y - expected_y) < 0.01

    def test_fourbar_simulation(self):
        """Test that built mechanism can be simulated."""
        mechanism = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .build()
        )

        # Should be able to step through simulation
        positions = list(mechanism.step())
        assert len(positions) > 0

        # Each position should have coordinates for all joints
        for pos in positions:
            assert len(pos) == len(mechanism.joints)


class TestSliderCrankAssembly:
    """Tests for slider-crank mechanism assembly."""

    def test_slider_crank(self):
        """Test assembling a slider-crank mechanism."""
        mechanism = (
            MechanismBuilder("slider-crank")
            .add_ground_link("ground", ports={"O": (0, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O", omega=0.1)
            .add_link("rod", length=3.0)
            .add_slide_axis("rail", through=(0, 0), direction=(1, 0))
            .connect("crank.tip", "rod.0")
            .connect_prismatic("rod.1", "rail")
            .build()
        )

        assert isinstance(mechanism, Mechanism)
        assert mechanism.name == "slider-crank"

        # Should have prismatic joint
        prismatic_found = False
        for joint in mechanism.joints:
            from pylinkage.mechanism import PrismaticJoint

            if isinstance(joint, PrismaticJoint):
                prismatic_found = True
                break
        assert prismatic_found


class TestTernaryLinkAssembly:
    """Tests for ternary link assembly."""

    def test_ternary_link_creation(self):
        """Test adding a ternary link."""
        builder = MechanismBuilder("test")
        builder.add_ternary_link("coupler", port_geometry={"A": (0, 0), "B": (3, 0), "P": (1.5, 1)})
        assert "coupler" in builder._pending_links
        link = builder._pending_links["coupler"]
        assert len(link.ports) == 3

    def test_ternary_link_distances(self):
        """Test that ternary link computes distances correctly."""
        builder = MechanismBuilder("test")
        builder.add_ternary_link("coupler", port_geometry={"A": (0, 0), "B": (3, 0), "P": (0, 4)})
        link = builder._pending_links["coupler"]

        # A to B should be 3
        assert link.get_port_distance("A", "B") == 3.0
        # A to P should be 4
        assert link.get_port_distance("A", "P") == 4.0
        # B to P should be 5 (3-4-5 triangle)
        assert link.get_port_distance("B", "P") == 5.0

    def test_ternary_link_invalid_port_count(self):
        """Test that ternary link requires exactly 3 ports."""
        builder = MechanismBuilder("test")
        with pytest.raises(ValueError, match="exactly 3 ports"):
            builder.add_ternary_link("coupler", port_geometry={"A": (0, 0), "B": (3, 0)})


class TestValidation:
    """Tests for builder validation."""

    def test_missing_ground_link(self):
        """Test validation fails without ground link."""
        builder = MechanismBuilder("test")
        builder.add_link("link", length=1.0)

        with pytest.raises(ValueError, match="No ground link"):
            builder.build()

    def test_invalid_connection_port(self):
        """Test validation fails with invalid port reference."""
        builder = (
            MechanismBuilder("test")
            .add_ground_link("ground", ports={"O1": (0, 0)})
            .add_link("link", length=1.0)
            .connect("link.0", "nonexistent.port")
        )

        with pytest.raises(ValueError, match="Unknown port"):
            builder.build()

    def test_invalid_driver_motor_port(self):
        """Test validation fails when driver references invalid ground port."""
        builder = (
            MechanismBuilder("test")
            .add_ground_link("ground", ports={"O1": (0, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O2")
        )

        with pytest.raises(ValueError, match="unknown ground port"):
            builder.build()

    def test_invalid_slide_axis(self):
        """Test validation fails with invalid slide axis reference."""
        builder = (
            MechanismBuilder("test")
            .add_ground_link("ground", ports={"O": (0, 0)})
            .add_link("link", length=1.0)
            .connect_prismatic("link.0", "nonexistent_axis")
        )

        with pytest.raises(ValueError, match="Unknown slide axis"):
            builder.build()


class TestAssemblyErrors:
    """Tests for assembly error handling."""

    def test_unbuildable_geometry(self):
        """Test that unbuildable geometry raises error."""
        # Links too short to reach
        builder = (
            MechanismBuilder("test")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (10, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=1.0)  # Too short
            .add_link("rocker", length=1.0)  # Too short
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
        )

        with pytest.raises(UnbuildableError):
            builder.build()

    def test_underconstrained_system(self):
        """Test that underconstrained system raises error."""
        builder = (
            MechanismBuilder("test")
            .add_ground_link("ground", ports={"O1": (0, 0)})
            .add_link("floating", length=1.0)
            # Link not connected to anything
        )

        with pytest.raises(UnderconstrainedError):
            builder.build()


class TestBranchSelection:
    """Tests for assembly branch selection."""

    def test_default_branch(self):
        """Test that default branch (0) is used."""
        mechanism = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .build()
        )

        # Get the coupler-rocker joint position
        coupler_rocker_y = None
        for joint in mechanism.joints:
            if "coupler" in joint.id and "rocker" in joint.id:
                coupler_rocker_y = joint.y
                break

        assert coupler_rocker_y is not None

    def test_alternate_branch(self):
        """Test selecting alternate branch."""
        # Build with default branch
        mech_default = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .build()
        )

        # Build with alternate branch for coupler.1
        mech_alt = (
            MechanismBuilder("four-bar")
            .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
            .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
            .add_link("coupler", length=3.5)
            .add_link("rocker", length=3.0)
            .connect("crank.tip", "coupler.0")
            .connect("coupler.1", "rocker.0")
            .connect("rocker.1", "ground.O2")
            .set_branch("coupler.1", 1)
            .build()
        )

        # The two mechanisms should have different configurations
        # (specifically, the coupler.1 joint should be in different positions)
        def get_joint_positions(mech):
            return {j.id: (j.x, j.y) for j in mech.joints}

        pos_default = get_joint_positions(mech_default)
        pos_alt = get_joint_positions(mech_alt)

        # At least one joint should have a different position
        # (The rocker.0/coupler.1 joint should differ)
        positions_differ = False
        for joint_id in pos_default:
            if joint_id in pos_alt:
                d_x, d_y = pos_default[joint_id]
                a_x, a_y = pos_alt[joint_id]
                if (
                    d_x is not None and a_x is not None
                    and (abs(d_x - a_x) > 0.01 or abs(d_y - a_y) > 0.01)
                ):
                        positions_differ = True
                        break

        assert positions_differ, "Branch selection should produce different configurations"


class TestQuaternaryLink:
    """Tests for quaternary link."""

    def test_quaternary_link_creation(self):
        """Test adding a quaternary link."""
        builder = MechanismBuilder("test")
        builder.add_quaternary_link(
            "plate", port_geometry={"A": (0, 0), "B": (2, 0), "C": (2, 1), "D": (0, 1)}
        )
        assert "plate" in builder._pending_links
        link = builder._pending_links["plate"]
        assert len(link.ports) == 4

    def test_quaternary_link_invalid_port_count(self):
        """Test that quaternary link requires exactly 4 ports."""
        builder = MechanismBuilder("test")
        with pytest.raises(ValueError, match="exactly 4 ports"):
            builder.add_quaternary_link(
                "plate", port_geometry={"A": (0, 0), "B": (2, 0), "C": (2, 1)}
            )
