"""Tests for hypergraph/conversion.py (topology + dimensions separation)."""

import unittest

import pylinkage as pl
from pylinkage.hypergraph._types import JointType, NodeRole
from pylinkage.hypergraph.conversion import from_linkage


class TestFromLinkage(unittest.TestCase):
    """Tests for from_linkage conversion."""

    def test_simple_fourbar(self):
        """Test conversion of simple four-bar linkage to hypergraph."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.1, name="crank")
        pin = pl.Revolute(2, 1, joint0=crank, joint1=(3, 0), distance0=2, distance1=2, name="pin")
        linkage = pl.Linkage(
            joints=(ground, crank, pin),
            order=(ground, crank, pin),
            name="Four-bar",
        )

        hg, dims = from_linkage(linkage)

        # Verify hypergraph structure
        self.assertEqual(hg.name, "Four-bar")
        self.assertGreater(len(hg.nodes), 0)
        self.assertGreater(len(hg.edges), 0)

        # Check that ground node exists
        ground_nodes = [n for n in hg.nodes.values() if n.role == NodeRole.GROUND]
        self.assertGreater(len(ground_nodes), 0)

        # Check that driver node exists
        driver_nodes = [n for n in hg.nodes.values() if n.role == NodeRole.DRIVER]
        self.assertEqual(len(driver_nodes), 1)

        # Check dimensions has positions
        self.assertGreater(len(dims.node_positions), 0)

    def test_with_fixed_joint(self):
        """Test conversion with Fixed joint."""
        ground = pl.Static(0, 0, name="ground")
        ref = pl.Static(1, 0, name="ref")
        fixed = pl.Fixed(joint0=ground, joint1=ref, distance=1.5, angle=0.5, name="fixed")

        linkage = pl.Linkage(
            joints=(ground, ref, fixed),
            order=(ground, ref, fixed),
            name="Fixed Test",
        )

        hg, dims = from_linkage(linkage)

        # Should have edges connecting fixed to its parents
        self.assertGreater(len(hg.edges), 0)

    def test_with_prismatic_joint(self):
        """Test conversion with Prismatic joint."""
        ground = pl.Static(0, 0, name="ground")
        line_start = pl.Static(0, 2, name="line_start")
        line_end = pl.Static(5, 2, name="line_end")
        prismatic = pl.Prismatic(
            2,
            2,
            joint0=ground,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2.5,
            name="prismatic",
        )

        linkage = pl.Linkage(
            joints=(ground, line_start, line_end, prismatic),
            order=(ground, line_start, line_end, prismatic),
            name="Prismatic Test",
        )

        hg, dims = from_linkage(linkage)

        # Check prismatic joint was converted correctly
        prismatic_nodes = [n for n in hg.nodes.values() if n.joint_type == JointType.PRISMATIC]
        self.assertEqual(len(prismatic_nodes), 1)

    def test_preserves_positions_in_dimensions(self):
        """Test that joint positions are preserved in dimensions."""
        ground = pl.Static(5.5, 3.2, name="ground")
        crank = pl.Crank(7.1, 4.8, joint0=ground, distance=2, angle=0.3, name="crank")

        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        hg, dims = from_linkage(linkage)

        # Check positions are stored in dimensions
        self.assertIn("ground", dims.node_positions)
        self.assertEqual(dims.node_positions["ground"], (5.5, 3.2))

    def test_preserves_crank_angle_in_dimensions(self):
        """Test that crank angle is preserved in dimensions."""
        ground = pl.Static(0, 0, name="ground")
        crank = pl.Crank(1, 0, joint0=ground, distance=1, angle=0.42, name="crank")

        linkage = pl.Linkage(
            joints=(ground, crank),
            order=(ground, crank),
        )

        hg, dims = from_linkage(linkage)

        # Find crank in dimensions and check initial angle
        self.assertIn("crank", dims.driver_angles)
        self.assertEqual(dims.driver_angles["crank"].initial_angle, 0.42)

    def test_handles_unique_node_ids(self):
        """Test that duplicate names get unique IDs."""
        # Create joints with same name
        ground1 = pl.Static(0, 0, name="joint")
        ground2 = pl.Static(1, 0, name="joint")

        linkage = pl.Linkage(
            joints=(ground1, ground2),
            order=(ground1, ground2),
        )

        hg, dims = from_linkage(linkage)

        # All node IDs should be unique
        node_ids = list(hg.nodes.keys())
        self.assertEqual(len(node_ids), len(set(node_ids)))


if __name__ == "__main__":
    unittest.main()
