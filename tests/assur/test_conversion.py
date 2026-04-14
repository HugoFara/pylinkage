"""Tests for the conversion module (topology + dimensions separation)."""

import pylinkage as pl
from pylinkage.assur import (
    NodeRole,
    linkage_to_graph,
)


class TestLinkageToGraph:
    """Tests for the linkage_to_graph function."""

    def test_convert_simple_crank(self):
        """Test converting a simple crank to graph."""
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="crank")
        linkage = pl.Linkage(joints=[crank], order=[crank])

        graph, dimensions = linkage_to_graph(linkage)

        # Should have 2 nodes: ground anchor and crank
        assert len(graph.nodes) == 2

        # Check crank node
        crank_node = graph.nodes.get("crank")
        assert crank_node is not None
        assert crank_node.role == NodeRole.DRIVER

        # Check driver angle stored in dimensions
        driver_angle = dimensions.get_driver_angle("crank")
        assert driver_angle is not None
        assert abs(driver_angle.initial_angle - 0.31) < 0.01

        # Check edge exists
        assert len(graph.edges) == 1

    def test_convert_four_bar(self):
        """Test converting a four-bar linkage to graph."""
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="B")
        pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="C")
        linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin], name="Four-bar")

        graph, dimensions = linkage_to_graph(linkage)

        assert graph.name == "Four-bar"

        # Should have nodes for: crank anchor, crank, pin anchor, pin
        # Note: tuple anchors become implicit ground nodes
        assert len(graph.nodes) >= 2

        # Check edges
        assert len(graph.edges) >= 2

        # Check dimensions has positions
        assert len(dimensions.node_positions) > 0

    def test_convert_with_explicit_static(self):
        """Test converting linkage with explicit Static joints."""
        anchor0 = pl.Static(0, 0, name="anchor0")
        anchor1 = pl.Static(3, 0, name="anchor1")
        crank = pl.Crank(0, 1, joint0=anchor0, angle=0.31, distance=1, name="crank")
        pin = pl.Revolute(3, 2, joint0=crank, joint1=anchor1, distance0=3, distance1=1, name="pin")
        linkage = pl.Linkage(joints=[anchor0, anchor1, crank, pin], order=[crank, pin])

        graph, dimensions = linkage_to_graph(linkage)

        # Check Static joints become GROUND nodes
        assert graph.nodes["anchor0"].role == NodeRole.GROUND
        assert graph.nodes["anchor1"].role == NodeRole.GROUND

        # Check Crank becomes DRIVER
        assert graph.nodes["crank"].role == NodeRole.DRIVER

        # Check Revolute becomes DRIVEN
        assert graph.nodes["pin"].role == NodeRole.DRIVEN

        # Check positions stored in dimensions
        assert "anchor0" in dimensions.node_positions
        assert "anchor1" in dimensions.node_positions


class TestLinkageToGraphFixed:
    """Tests for linkage_to_graph with Fixed joints."""

    def test_convert_with_fixed_joint(self):
        """Test converting linkage with Fixed joint."""
        anchor0 = pl.Static(0, 0, name="anchor0")
        anchor1 = pl.Static(3, 0, name="anchor1")
        fixed = pl.Fixed(joint0=anchor0, joint1=anchor1, distance=2.0, angle=0.5, name="fixed")
        linkage = pl.Linkage(joints=[anchor0, anchor1, fixed], order=[anchor0, anchor1, fixed])

        graph, dimensions = linkage_to_graph(linkage)

        # Check Fixed becomes DRIVEN (deterministic position)
        assert graph.nodes["fixed"].role == NodeRole.DRIVEN


class TestLinkageToGraphPrismatic:
    """Tests for linkage_to_graph with Prismatic joints."""

    def test_convert_with_prismatic_joint(self):
        """Test converting linkage with Prismatic joint."""
        anchor = pl.Static(0, 0, name="anchor")
        line_start = pl.Static(0, 2, name="line_start")
        line_end = pl.Static(5, 2, name="line_end")
        prismatic = pl.Prismatic(
            2,
            2,
            joint0=anchor,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2.5,
            name="prismatic",
        )
        linkage = pl.Linkage(
            joints=[anchor, line_start, line_end, prismatic],
            order=[anchor, line_start, line_end, prismatic],
        )

        graph, dimensions = linkage_to_graph(linkage)

        # Check Prismatic becomes DRIVEN
        assert graph.nodes["prismatic"].role == NodeRole.DRIVEN
