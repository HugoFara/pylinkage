"""Tests for the conversion module (topology + dimensions separation)."""

import pylinkage as pl
from pylinkage.assur import (
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    graph_to_linkage,
    linkage_to_graph,
)
from pylinkage.dimensions import Dimensions


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


class TestGraphToLinkage:
    """Tests for the graph_to_linkage function."""

    def test_convert_simple_graph(self):
        """Test converting a simple graph to linkage."""
        # Topology only
        graph = LinkageGraph(name="Four-bar")

        # Ground points
        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("D", role=NodeRole.GROUND))

        # Driver
        graph.add_node(Node("B", role=NodeRole.DRIVER))

        # Driven
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        # Edges
        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BC", source="B", target="C"))
        graph.add_edge(Edge("CD", source="C", target="D"))

        # Dimensions separate
        from pylinkage.dimensions import DriverAngle

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "D": (3.0, 0.0),
                "B": (0.0, 1.0),
                "C": (3.0, 2.0),
            },
            driver_angles={"B": DriverAngle(angular_velocity=0.1, initial_angle=0.31)},
            edge_distances={"AB": 1.0, "BC": 3.0, "CD": 1.0},
        )

        linkage = graph_to_linkage(graph, dimensions)

        assert linkage.name == "Four-bar"

        # Check we have the right joint types
        joint_names = [j.name for j in linkage.joints]
        assert "A" in joint_names or "D" in joint_names  # At least one ground

    def test_roundtrip_conversion(self):
        """Test that linkage -> graph -> linkage preserves structure."""
        # Create original linkage
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="B")
        pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="C")
        original = pl.Linkage(joints=[crank, pin], order=[crank, pin], name="Test")

        # Convert to graph (returns topology + dimensions)
        graph, dimensions = linkage_to_graph(original)

        # Convert back to linkage (requires dimensions)
        restored = graph_to_linkage(graph, dimensions)

        # Check name preserved
        assert restored.name == original.name

        # Check joint count (may differ due to implicit anchors becoming explicit)
        assert len(restored.joints) >= len(original.joints)

    def test_converted_linkage_can_simulate(self):
        """Test that converted linkage can run simulation."""
        # Topology only
        graph = LinkageGraph(name="Four-bar")

        graph.add_node(Node("A", role=NodeRole.GROUND))
        graph.add_node(Node("D", role=NodeRole.GROUND))
        graph.add_node(Node("B", role=NodeRole.DRIVER))
        graph.add_node(Node("C", role=NodeRole.DRIVEN))

        graph.add_edge(Edge("AB", source="A", target="B"))
        graph.add_edge(Edge("BC", source="B", target="C"))
        graph.add_edge(Edge("CD", source="C", target="D"))

        # Dimensions separate
        from pylinkage.dimensions import DriverAngle

        dimensions = Dimensions(
            node_positions={
                "A": (0.0, 0.0),
                "D": (3.0, 0.0),
                "B": (0.0, 1.0),
                "C": (3.0, 2.0),
            },
            driver_angles={"B": DriverAngle(angular_velocity=0.1, initial_angle=0.31)},
            edge_distances={"AB": 1.0, "BC": 3.0, "CD": 1.0},
        )

        linkage = graph_to_linkage(graph, dimensions)

        # Should be able to step without error
        step_count = 0
        for coords in linkage.step(iterations=10):
            step_count += 1
            assert len(coords) == len(linkage.joints)

        assert step_count == 10


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
